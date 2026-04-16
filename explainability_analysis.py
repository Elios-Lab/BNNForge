"""
Explainability analysis: SHAP-based model comparison (FP vs BNN).
Supports EPD, ToF, AR, SC, Z24 datasets with TFLite inference.
"""

import os, warnings, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import interp1d
import shap
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

try:
    import larq
    HAS_LARQ = True
except ImportError:
    HAS_LARQ = False

DPI = 150


# DATA LOADING

def load_dataset(data_dir):
    import pandas as pd
    ds = {}

    x = np.load(f'{data_dir}/AR/AR_test_X.npy').astype(np.float32)
    y = np.load(f'{data_dir}/AR/AR_test_y.npy').flatten()
    if x.ndim == 1: x = x[None, :, None]
    elif x.ndim == 2 and x.shape[-1] != 1: x = x[:, :, None]
    elif x.ndim == 2: x = x[None]
    ds['AR'] = (x, y)

    epd_npy = f'{data_dir}/EPD/EPD_test_data.npy'
    epd_lbl_npy = f'{data_dir}/EPD/EPD_test_label.npy'
    epd_lbl_txt = f'{data_dir}/EPD/EPD_label.txt'
    if os.path.exists(epd_npy):
        x = np.load(epd_npy).astype(np.float32)
        y = np.load(epd_lbl_npy).flatten()
        if x.ndim == 2: x = x[None, :, :, None]
        elif x.ndim == 3: x = x[:, :, :, None]
        x_rgb = None
    else:
        import glob as _glob
        img_files = sorted(_glob.glob(f'{data_dir}/EPD/EPD_test_data.*'))
        img_files = [f for f in img_files if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
        from PIL import Image as _Image
        img_rgb  = _Image.open(img_files[0]).convert('RGB')
        img_gray = img_rgb.convert('L')
        x_rgb = np.array(img_rgb, dtype=np.float32) / 255.0
        x = np.array(img_gray, dtype=np.float32) / 255.0
        x = x[None, :, :, None]
        with open(epd_lbl_txt) as f:
            y = np.array([int(f.read().strip())])
    ds['EPD'] = (x, y)
    ds['EPD_rgb'] = x_rgb   # None or (H, W, 3) float32

    x = np.load(f'{data_dir}/SC/SC_test_X.npy').astype(np.float32)
    y = np.load(f'{data_dir}/SC/SC_test_y.npy').flatten()
    if x.ndim == 2: x = x[None, :, :, None]
    elif x.ndim == 3: x = x[:, :, :, None]
    ds['SC'] = (x, y)

    tof = pd.read_csv(f'{data_dir}/ToF/Tof_sample.csv',
                      header=None).values.astype(np.float32)
    with open(f'{data_dir}/ToF/ToF_label.txt') as f:
        tof_y = np.array([int(f.read().strip())])
    ds['ToF'] = (tof[None, :, :, None], tof_y)

    x = joblib.load(f'{data_dir}/Z24/z24_data.pkl').astype(np.float32)
    y = np.array([float(joblib.load(f'{data_dir}/Z24/z24_label.pkl'))])
    if x.ndim == 1: x = x[None, :, None]
    ds['Z24'] = (x, y)

    return ds


def normalize_01(x, gamma=0.4):
    mn, mx = x.min(), x.max()
    normed = (x - mn) / (mx - mn + 1e-8)
    return np.power(normed, gamma)


def squeeze_ch(x):
    if x.ndim >= 1 and x.shape[-1] == 1:
        return x[..., 0]
    return x


# MODEL LOADING

def load_keras(path):
    """Load Keras model (.h5). Tries larq context if available."""
    if HAS_LARQ:
        try:
            with larq.context.metrics_scope():
                return tf.keras.models.load_model(path, compile=False)
        except Exception:
            pass
    return tf.keras.models.load_model(path, compile=False)


def load_tflite(path):
    # Try standard TFLite first (FP models and any model without custom ops)
    try:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        interp._is_lce = False
        return interp
    except RuntimeError:
        pass
    # Fall back to LCE interpreter for BNN models with LceQuantize ops
    from larq_compute_engine.tflite.python.interpreter import Interpreter as LceInterpreter
    with open(path, 'rb') as f:
        model_bytes = f.read()
    interp = LceInterpreter(model_bytes)
    interp._is_lce = True
    return interp


def tflite_predict(interp, X):
    """Batched TFLite inference. X: (N, …)"""
    if getattr(interp, '_is_lce', False):
        exp = tuple(interp.input_shapes[0])   # e.g. (1, 101, 39) or (1, 30, 30, 1)
        if X.shape[1:] != exp[1:]:
            X = X.reshape((X.shape[0],) + exp[1:])
        return interp.predict(X)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    model_shape = tuple(inp['shape'])   # e.g. (1, 101, 39) or (1, 30, 30, 1)
    res = []
    for i in range(X.shape[0]):
        x_slice = X[i:i+1].astype(inp['dtype'])
        if x_slice.shape != model_shape:
            x_slice = x_slice.reshape(model_shape)
        interp.set_tensor(inp['index'], x_slice)
        interp.invoke()
        res.append(interp.get_tensor(out['index']).copy())
    return np.concatenate(res, axis=0)


def make_tflite_fn(interp):
    return lambda X: tflite_predict(interp, X)


def make_keras_fn(model):
    return lambda X: model(X, training=False).numpy()


# EXPLAINABILITY


def _signed_power(x, gamma=0.4):
    """Power stretch preserving sign."""
    return np.sign(x) * np.power(np.abs(x), gamma)


def _shap_rgba(sv, cmap_name='RdBu_r', alpha_gamma=1.0):
    """Convert signed SHAP map to RGBA with color (blue=neg, red=pos) and alpha."""
    max_abs = max(np.abs(sv).max(), 1e-8)
    norm = Normalize(vmin=-max_abs, vmax=max_abs)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm(sv))                                # color from raw values
    sv_s = _signed_power(sv)
    alpha = (np.abs(sv_s) / max(np.abs(sv_s).max(), 1e-8)) ** alpha_gamma
    rgba[..., 3] = np.clip(alpha, 0.0, 0.7)
    return rgba


def _extract_signed_shap(sv, n_seg, class_idx):
    if isinstance(sv, list):
        if len(sv) > class_idx:
            return np.array(sv[class_idx][0], dtype=np.float32)
        return np.mean([np.array(v[0]) for v in sv], axis=0).astype(np.float32)
    sv_arr = np.array(sv)
    if sv_arr.ndim == 3 and sv_arr.shape[0] > class_idx and sv_arr.shape[2] == n_seg:
        return sv_arr[class_idx, 0].astype(np.float32)
    # Find the axis with n_seg elements
    seg_ax = next((i for i, d in enumerate(sv_arr.shape) if d == n_seg), None)
    if seg_ax is not None:
        other = tuple(i for i in range(sv_arr.ndim) if i != seg_ax)
        vals  = sv_arr.mean(axis=other).flatten() if other else sv_arr.flatten()
    else:
        vals = sv_arr.flatten()
    if len(vals) > n_seg:
        vals = vals[:n_seg]
    elif len(vals) < n_seg:
        vals = np.pad(vals, (0, n_seg - len(vals)))
    return vals.astype(np.float32)


def _is_2d_image(x):
    if x.ndim == 2 and min(x.shape) > 1:
        return True
    if x.ndim == 3 and x.shape[2] <= 4 and min(x.shape[:2]) > 1:
        return True
    return False


def segment_shap(predict_fn, x, background, n_segments=50, n_bg=50, nsamples=1000):
    """Segmented KernelSHAP for 2D images (patches) and 1D signals (chunks)."""
    orig  = x.shape

    if _is_2d_image(x):
        rows, cols = x.shape[0], x.shape[1]
        n_row_segs = max(3, int(round(np.sqrt(n_segments * rows / (cols + 1e-8)))))
        n_col_segs = max(3, int(round(np.sqrt(n_segments * cols / (rows + 1e-8)))))
        row_splits = np.array_split(np.arange(rows), n_row_segs)
        col_splits = np.array_split(np.arange(cols), n_col_segs)
        patches  = [(ri, ci) for ri in row_splits for ci in col_splits]
        n_p      = len(patches)

        if x_range <= 1.0:
            fill_vals = [0.5] * n_p
        else:
            fill_vals = [float(x[ri[:, None], ci[None, :], 0].mean()
                               if x.ndim == 3 else x[np.ix_(ri, ci)].mean())
                         for ri, ci in patches]

        def mp2d(mask_batch):
            results = []
            for mask in mask_batch:
                xm = x.copy()
                for si, (ri, ci) in enumerate(patches):
                    if mask[si] < 0.5:
                        if x.ndim == 3:
                            xm[ri[:, None], ci[None, :], :] = fill_vals[si]
                        else:
                            xm[np.ix_(ri, ci)] = fill_vals[si]
                results.append(predict_fn(xm[np.newaxis])[0])
            return np.array(results)

        base_out  = mp2d(np.ones((1, n_p)))
        class_idx = int(np.argmax(base_out[0]))
        expl = shap.KernelExplainer(mp2d, np.zeros((1, n_p), dtype=np.float32))
        sv   = expl.shap_values(np.ones((1, n_p), dtype=np.float32),
                                 nsamples=nsamples, silent=True)
        seg_vals = _extract_signed_shap(sv, n_p, class_idx)

        imp_map = np.zeros(orig, dtype=np.float32)
        for si, (ri, ci) in enumerate(patches):
            if orig[-1] == 3 if len(orig) == 3 else False:
                imp_map[ri[:, None], ci[None, :], :] = seg_vals[si]
            else:
                imp_map[np.ix_(ri, ci)] = seg_vals[si]
        return imp_map

    flat = x.flatten().astype(np.float32)
    n    = len(flat)
    segs = np.array_split(np.arange(n), n_segments)
    seg_means = np.array([flat[seg].mean() for seg in segs], dtype=np.float32)
    bg_1d = np.concatenate([np.full(len(seg), seg_means[si])
                             for si, seg in enumerate(segs)])

    def mp1d(mask_batch):
        results = []
        for mask in mask_batch:
            xm = bg_1d.copy()
            for si, seg in enumerate(segs):
                if mask[si] > 0.5:
                    xm[seg] = flat[seg]
            results.append(predict_fn(xm.reshape((1,) + orig))[0])
        return np.array(results)

    base_out  = mp1d(np.ones((1, n_segments)))
    class_idx = int(np.argmax(base_out[0]))
    expl = shap.KernelExplainer(mp1d, np.zeros((1, n_segments), dtype=np.float32))
    sv   = expl.shap_values(np.ones((1, n_segments), dtype=np.float32),
                             nsamples=nsamples, silent=True)
    seg_vals = _extract_signed_shap(sv, n_segments, class_idx)

    if np.max(np.abs(seg_vals)) < 1e-6:
        for si in range(n_segments):
            mask = np.ones(n_segments, dtype=np.float32)
            mask[si] = 0.0
            occ_out = mp1d(mask[np.newaxis])[0]
            seg_vals[si] = float(base_out[0][class_idx] - occ_out[class_idx])

    full_sal = np.zeros(n, dtype=np.float32)
    for si, seg in enumerate(segs):
        full_sal[seg] = seg_vals[si]
    return full_sal.reshape(orig)





def kernel_shap(predict_fn, x, background, n_bg=50, nsamples=1000):
    """KernelSHAP with automatic segmentation for high-dim inputs (>200 features)."""
    orig      = x.shape
    n_features = int(np.prod(orig))

    if n_features > 200:
        n_seg = min(200, max(50, int(np.sqrt(n_features) * 3)))
        return segment_shap(predict_fn, x, background,
                             n_segments=n_seg, n_bg=n_bg, nsamples=nsamples)

    nsamples = max(nsamples, n_features * 10)

    n_bg_actual = min(n_bg, len(background))
    bg = background[:n_bg_actual].reshape(n_bg_actual, -1).astype(np.float32)
    if n_bg_actual < 5:
        n_synth = max(5, n_bg) - n_bg_actual
        bg = np.vstack([bg, np.zeros((n_synth, bg.shape[1]), dtype=np.float32)])

    def flat_pred(xb):
        return predict_fn(xb.reshape((-1,) + orig))

    expl      = shap.KernelExplainer(flat_pred, bg)
    sv        = expl.shap_values(x.flatten()[np.newaxis],
                                  nsamples=nsamples, silent=True)
    base_out  = predict_fn(x[np.newaxis])[0]
    class_idx = int(np.argmax(base_out))
    seg_vals  = _extract_signed_shap(sv, n_features, class_idx)
    return seg_vals.reshape(orig)


def deep_shap(model, x, background, n_bg=50):
    """DeepSHAP for Keras models."""
    n_bg_actual = min(n_bg, len(background))
    bg_data = background[:n_bg_actual].astype(np.float32)
    if n_bg_actual < 5:
        n_synth = max(5, n_bg) - n_bg_actual
        bg_data = np.concatenate(
            [bg_data, np.zeros((n_synth,) + background.shape[1:], dtype=np.float32)],
            axis=0)
    expl = shap.DeepExplainer(model, bg_data)
    sv   = expl.shap_values(x[np.newaxis])
    if isinstance(sv, list):
        sv = np.mean([np.abs(v[0]) for v in sv], axis=0)
    else:
        sv = np.abs(sv[0])
        n_features = x.size
        if sv.size != n_features and sv.size % n_features == 0:
            sv = sv.reshape(-1, n_features).mean(axis=0).reshape(x.shape)
    return normalize_01(sv)


# PLOTTING

def plot_2d_shap(x2d, fp_shap, bnn_shap, title, fp_label, bnn_label, save_path,
                 x_rgb=None, input_cmap='gray', neutral_bg=False):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    fig.suptitle(f"{title}   –   SHAP  (signed)", fontsize=13, fontweight='bold')

    disp = x_rgb if x_rgb is not None else x2d
    if x_rgb is not None:
        axes[0].imshow(disp, origin='lower', aspect='auto')
    else:
        img2d = x2d[..., 0] if x2d.ndim == 3 else x2d
        axes[0].imshow(img2d, cmap=input_cmap, origin='lower', aspect='auto')
    axes[0].set_title('Input')
    blur_sigma = max(0.8, min(x2d.shape[:2]) / 15)
    fp_sv   = gaussian_filter(fp_shap,  sigma=blur_sigma)
    bnn_sv  = gaussian_filter(bnn_shap, sigma=blur_sigma)
    fp_max_abs = max(np.abs(fp_sv).max(), 1e-8)
    bnn_max_abs = max(np.abs(bnn_sv).max(), 1e-8)
    fp_sv = fp_sv / fp_max_abs
    bnn_sv = bnn_sv / bnn_max_abs
    norm_div = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)

    for ax, sv, lab in [(axes[1], fp_sv,  fp_label),
                         (axes[2], bnn_sv, bnn_label)]:
        if neutral_bg:
            if x_rgb is not None:
                gray_bg = 0.299 * x_rgb[..., 0] + 0.587 * x_rgb[..., 1] + 0.114 * x_rgb[..., 2]
            else:
                gray_bg = x2d[..., 0] if x2d.ndim == 3 else x2d
            ax.imshow(gray_bg, cmap='gray', origin='lower', aspect='auto')
        else:
            if x_rgb is not None:
                ax.imshow(x_rgb, origin='lower', aspect='auto')
            else:
                img2d = x2d[..., 0] if x2d.ndim == 3 else x2d
                ax.imshow(img2d, cmap=input_cmap, origin='lower', aspect='auto')
        ax.imshow(_shap_rgba(sv), origin='lower', aspect='auto',
                  interpolation='bilinear')
        ax.set_title(lab, fontsize=10)
        im = plt.cm.ScalarMappable(norm=norm_div, cmap='RdBu_r')
        fig.colorbar(im, ax=ax, shrink=0.85, label='SHAP value')
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def plot_1d_shap(x1d, fp_shap, bnn_shap, title, fp_label, bnn_label,
                 save_path, xlabel='Sample index'):
    n = len(x1d); t = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                              constrained_layout=True)
    fig.suptitle(f"{title}   –   SHAP  (signed)", fontsize=13, fontweight='bold')
    fp_shap  = _signed_power(fp_shap)
    bnn_shap = _signed_power(bnn_shap)
    fp_max_abs = max(np.abs(fp_shap).max(), 1e-8)
    bnn_max_abs = max(np.abs(bnn_shap).max(), 1e-8)
    fp_shap = fp_shap / fp_max_abs
    bnn_shap = bnn_shap / bnn_max_abs
    norm_sv  = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    for ax, sv, lab in [(axes[0], fp_shap,  fp_label),
                         (axes[1], bnn_shap, bnn_label)]:
        pts  = np.array([t, x1d]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap='RdBu_r', norm=norm_sv, lw=0.8, zorder=3)
        lc.set_array((sv[:-1] + sv[1:]) / 2)
        ax.add_collection(lc)
        ax.set_xlim(t[0], t[-1])
        pad = (x1d.max() - x1d.min()) * 0.05
        ax.set_ylim(x1d.min() - pad, x1d.max() + pad)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.set_title(lab, fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_sv, cmap='RdBu_r'),
                 ax=axes[:], shrink=0.6, label='SHAP value (blue=neg, red=pos)')
    axes[-1].set_xlabel(xlabel, fontsize=9)
    fig.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)



# TASK RUNNERS

def _pred_class(predict_fn, x):
    return int(np.argmax(predict_fn(x[np.newaxis])[0]))


def _label_str(pred, true):
    if true is None: return str(pred)
    ok = '✓' if pred == true else f'✗(true={true})'
    return f"{pred}{ok}"

def run_2d_task(task, x_s, fp_interp, bnn_interp, background,
                out_dir, idx, n_bg, true_label=None, x_rgb=None):
    """2D image SHAP analysis (EPD, ToF)."""
    print(f"[{task}] sample={idx}")
    fp_fn  = make_tflite_fn(fp_interp)
    bnn_fn = make_tflite_fn(bnn_interp)

    fp_c  = _pred_class(fp_fn,  x_s)
    bnn_c = _pred_class(bnn_fn, x_s)
    print(f"  FP={_label_str(fp_c, true_label)}  BNN={_label_str(bnn_c, true_label)}")

    fp_shap  = squeeze_ch(kernel_shap(fp_fn,  x_s, background, n_bg))
    bnn_shap = squeeze_ch(kernel_shap(bnn_fn, x_s, background, n_bg))
    plot_2d_shap(x2d, fp_shap, bnn_shap, task,
                 'FP – KernelSHAP', 'BNN – KernelSHAP',
                 f'{out_dir}/{task}_s{idx}_shap.png', x_rgb=x_rgb)

def run_1d_task(task, x_s, fp_interp, bnn_interp, background,
                out_dir, idx, n_bg,
                xlabel='Sample index', ylabel='Amplitude', true_label=None):
    """1D signal SHAP analysis (AR)."""
    print(f"[{task}] sample={idx}")
    fp_fn  = make_tflite_fn(fp_interp)
    bnn_fn = make_tflite_fn(bnn_interp)

    fp_c  = _pred_class(fp_fn,  x_s)
    bnn_c = _pred_class(bnn_fn, x_s)
    print(f"  FP={_label_str(fp_c, true_label)}  BNN={_label_str(bnn_c, true_label)}")

    fp_shap  = squeeze_ch(segment_shap(fp_fn,  x_s, background, n_bg=n_bg)).flatten()
    bnn_shap = squeeze_ch(segment_shap(bnn_fn, x_s, background, n_bg=n_bg)).flatten()
    plot_1d_shap(x1d, fp_shap, bnn_shap, task,
                 'FP – SegmentSHAP', 'BNN – SegmentSHAP',
                 f'{out_dir}/{task}_s{idx}_shap.png', xlabel)


def run_z24_task(x_s, fp_interp, bnn_interp, background,
                 out_dir, idx, n_bg, true_label=None):
    """Long time-series SHAP analysis (Z24)."""
    task = 'Z24'
    print(f"[{task}] sample={idx}")
    bnn_fn = make_tflite_fn(bnn_interp)
    fp_fn = make_tflite_fn(fp_interp)

    fp_c  = _pred_class(fp_fn,  x_s)
    bnn_c = _pred_class(bnn_fn, x_s)
    print(f"  FP={_label_str(fp_c, true_label)}  BNN={_label_str(bnn_c, true_label)}")

    fp_shap = squeeze_ch(
        kernel_shap(fp_fn, x_s, background, n_bg=n_bg)).flatten()
    bnn_shap = squeeze_ch(
        kernel_shap(bnn_fn, x_s, background, n_bg=n_bg)).flatten()
    
    x1d = squeeze_ch(x_s).flatten()
    plot_1d_shap(x1d, fp_shap, bnn_shap, task,
                 'FP – KernelSHAP', 'BNN – KernelSHAP',
                 f'{out_dir}/{task}_s{idx}_shap.png',
                 xlabel='Time sample')


def run_sc_task(x_s, fp_interp, bnn_interp, background,
                out_dir, idx, n_bg, true_label=None):
    """Speech/audio MFCC SHAP analysis (SC)."""
    task = 'SC'
    print(f"[{task}] sample={idx}")
    fp_fn  = make_tflite_fn(fp_interp)
    bnn_fn = make_tflite_fn(bnn_interp)

    fp_c  = _pred_class(fp_fn,  x_s)
    bnn_c = _pred_class(bnn_fn, x_s)
    print(f"  FP={_label_str(fp_c, true_label)}  BNN={_label_str(bnn_c, true_label)}")

    fp_shap  = squeeze_ch(kernel_shap(fp_fn,  x_s, background, n_bg))
    bnn_shap = squeeze_ch(kernel_shap(bnn_fn, x_s, background, n_bg))
    x2d     = squeeze_ch(x_s)

    plot_2d_shap(x2d.T, fp_shap.T, bnn_shap.T, task,
                 'FP – KernelSHAP', 'BNN – KernelSHAP',
                 f'{out_dir}/{task}_s{idx}_shap.png',
                 input_cmap='magma', neutral_bg=True)


# MAIN

def main(data_dir='inputs', model_dir='.', out_dir='explainability_output',
         sample_idx=0, n_shap_bg=30):
    """
    Parameters
    ----------
    data_dir   : path to extracted inputs/ folder
    model_dir  : folder with all model files (.tflite)
    out_dir    : output directory for plots
    sample_idx : which sample index to explain
    n_shap_bg  : background samples for KernelSHAP
    """
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(data_dir)
    M  = lambda n: os.path.join(model_dir, n)

    print("\nEPD")
    epd_x, epd_y = ds['EPD']
    epd_rgb = ds.get('EPD_rgb')  # (H, W, 3) float32 or None
    run_2d_task('EPD', epd_x[sample_idx],
                load_tflite(M('EPD_fp.tflite')),
                load_tflite(M('EPD_bnn.tflite')),
                epd_x, out_dir, sample_idx, n_shap_bg,
                true_label=int(epd_y[sample_idx]) if sample_idx < len(epd_y) else None,
                x_rgb=epd_rgb)

    print("\nToF")
    tof_x, tof_y = ds['ToF']
    run_2d_task('ToF', tof_x[sample_idx],
                load_tflite(M('ToF_fp.tflite')),
                load_tflite(M('ToF_bnn.tflite')),
                tof_x, out_dir, sample_idx, n_shap_bg,
                true_label=int(tof_y[sample_idx]) if sample_idx < len(tof_y) else None)

    print("\nAR")
    ar_x, ar_y = ds['AR']
    run_1d_task('AR', ar_x[sample_idx],
                load_tflite(M('AR_fp.tflite')),
                load_tflite(M('AR_bnn.tflite')),
                ar_x, out_dir, sample_idx, n_shap_bg,
                true_label=int(ar_y[sample_idx]) if sample_idx < len(ar_y) else None)

    print("\nZ24")
    z24_x, z24_y = ds['Z24']
    run_z24_task(z24_x[sample_idx],
                 load_tflite(M('z24_model.tflite')),
                 load_tflite(M('z24_bnn.tflite')),
                 z24_x, out_dir, sample_idx, n_shap_bg,
                 true_label=int(z24_y[sample_idx]) if sample_idx < len(z24_y) else None)

    print("\nSC")
    sc_x, sc_y = ds['SC']
    run_sc_task(sc_x[sample_idx],
                load_tflite(M('SC_fp.tflite')),
                load_tflite(M('SC_bnn.tflite')),
                sc_x, out_dir, sample_idx, n_shap_bg,
                true_label=int(sc_y[sample_idx]) if sample_idx < len(sc_y) else None)

    print(f"\nDone. Output: {os.path.abspath(out_dir)}/")


# MULTI-SAMPLE RUNNER

def run_multi(data_dir, model_dir, out_dir, task, indices, n_shap_bg=30):
    """Run SHAP pipeline on multiple samples for a single task."""
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(data_dir)
    M  = lambda n: os.path.join(model_dir, n)

    cfg = {
        'EPD': dict(fp=M('EPD_fp.tflite'), bnn=M('EPD_bnn.tflite'),
                    is_2d=True,  is_sc=False, is_z24=False),
        'ToF': dict(fp=M('ToF_fp.tflite'), bnn=M('ToF_bnn.tflite'),
                    is_2d=True,  is_sc=False, is_z24=False),
        'AR':  dict(fp=M('AR_fp.tflite'),  bnn=M('AR_bnn.tflite'),
                    is_2d=False, is_sc=False, is_z24=False),
        'Z24': dict(fp=M('z24_model.tflite'), bnn=M('z24_bnn.tflite'),
                    is_2d=False, is_sc=False, is_z24=False),
        'SC':  dict(fp=M('SC_fp.tflite'),  bnn=M('SC_bnn.tflite'),
                    is_2d=False, is_sc=True,  is_z24=False),
    }
    c   = cfg[task]
    X, _ = ds[task]

    fp_tflite  = load_tflite(c['fp'])
    bnn_tflite = load_tflite(c['bnn'])

    for idx in np.array(indices):
        x_s = X[idx]
        if c['is_z24']:
            run_z24_task(x_s, fp_tflite, bnn_tflite, X, out_dir,
                         idx, n_shap_bg)
        elif c['is_sc']:
            run_sc_task(x_s, fp_tflite, bnn_tflite, X, out_dir,
                        idx, n_shap_bg)
        elif c['is_2d']:
            run_2d_task(task, x_s, fp_tflite, bnn_tflite, X, out_dir,
                        idx, n_shap_bg)
        else:
            run_1d_task(task, x_s, fp_tflite, bnn_tflite, X, out_dir,
                        idx, n_shap_bg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FP vs BNN explainability – SHAP')
    parser.add_argument('--data_dir',  default='inputs',
                        help='Path to extracted inputs/ folder')
    parser.add_argument('--model_dir', default='.',
                        help='Folder containing model files')
    parser.add_argument('--out_dir',   default='explainability_output',
                        help='Output folder for figures')
    parser.add_argument('--sample',    type=int, default=0,
                        help='Sample index to explain')
    parser.add_argument('--n_shap_bg', type=int, default=30,
                        help='Background samples for SHAP')
    args = parser.parse_args()

    main(data_dir   = args.data_dir,
         model_dir  = args.model_dir,
         out_dir    = args.out_dir,
         sample_idx = args.sample,
         n_shap_bg  = args.n_shap_bg)
