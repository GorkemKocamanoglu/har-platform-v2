// har_classifier.dart — on-device activity prediction from live HARNode data.
//
// Loads the forest exported by the training notebook (assets/har_model.json),
// keeps a rolling 2-second buffer per wrist, and reproduces the training
// notebook's 39 features exactly (same names, same math), so the forest sees
// the same inputs it was trained on.
//
// Usage from the app:
//   final clf = await HarClassifier.loadFromAsset('assets/har_model.json');
//   clf.addSample(deviceName, tMs, ax, ay, az, gx, gy, gz);   // every BLE packet
//   final label = clf.predict();                              // e.g. every 1s

import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart' show rootBundle;

class _Sample {
  final int tMs;
  final double ax, ay, az, gx, gy, gz;
  _Sample(this.tMs, this.ax, this.ay, this.az, this.gx, this.gy, this.gz);
}

class HarClassifier {
  final List<String> classes;
  final List<String> featureNames;
  final List<Map<String, dynamic>> trees;

  static const int windowMs = 2000;
  final Map<String, List<_Sample>> _buffers = {'L': [], 'R': []};

  HarClassifier(this.classes, this.featureNames, this.trees);

  static Future<HarClassifier> loadFromAsset(String path) async {
    final raw = json.decode(await rootBundle.loadString(path));
    return HarClassifier(
      List<String>.from(raw['classes']),
      List<String>.from(raw['feature_names']),
      List<Map<String, dynamic>>.from(raw['trees']),
    );
  }

  // ------------------------------------------------------------ ingest

  String? _tag(String deviceName) {
    final n = deviceName.toLowerCase();
    if (n.contains('left')) return 'L';
    if (n.contains('right')) return 'R';
    return null;
  }

  void addSample(String deviceName, int tMs, double ax, double ay, double az,
      double gx, double gy, double gz) {
    final tag = _tag(deviceName);
    if (tag == null) return;
    final buf = _buffers[tag]!;
    buf.add(_Sample(tMs, ax, ay, az, gx, gy, gz));
    final cutoff = tMs - windowMs;
    while (buf.isNotEmpty && buf.first.tMs < cutoff) {
      buf.removeAt(0);
    }
  }

  // ------------------------------------------------------------ stats helpers

  double _mean(List<double> v) => v.reduce((a, b) => a + b) / v.length;

  double _std(List<double> v) {
    final m = _mean(v);
    double s = 0;
    for (final x in v) s += (x - m) * (x - m);
    return math.sqrt(s / v.length); // population std, like numpy default
  }

  double _corr(List<double> a, List<double> b) {
    final n = math.min(a.length, b.length);
    if (n < 8) return 0.0;
    final aa = a.sublist(0, n), bb = b.sublist(0, n);
    final ma = _mean(aa), mb = _mean(bb);
    double num = 0, da = 0, db = 0;
    for (int i = 0; i < n; i++) {
      num += (aa[i] - ma) * (bb[i] - mb);
      da += (aa[i] - ma) * (aa[i] - ma);
      db += (bb[i] - mb) * (bb[i] - mb);
    }
    if (da < 1e-12 || db < 1e-12) return 0.0;
    return num / math.sqrt(da * db);
  }

  /// Dominant frequency + relative peak power, mirroring the training notebook:
  /// hanning window, DFT on the rfft grid, argmax within the 0.8–5.0 Hz band.
  List<double> _dominantFreq(List<double> sig, List<int> tsMs) {
    final n = sig.length;
    if (n < 8) return [0.0, 0.0];
    final durS = math.max((tsMs.last - tsMs.first) / 1000.0, 1e-6);
    final fs = n / durS;

    final m = _mean(sig);
    final x = List<double>.generate(
        n,
        (i) =>
            (sig[i] - m) *
            (0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)))); // hanning

    final nBins = n ~/ 2 + 1;
    final spec = List<double>.filled(nBins, 0.0);
    for (int k = 0; k < nBins; k++) {
      double re = 0, im = 0;
      for (int j = 0; j < n; j++) {
        final ang = -2 * math.pi * j * k / n;
        re += x[j] * math.cos(ang);
        im += x[j] * math.sin(ang);
      }
      spec[k] = math.sqrt(re * re + im * im);
    }

    double total = 0;
    for (int k = 1; k < nBins; k++) total += spec[k];
    total += 1e-9;

    int best = -1;
    for (int k = 0; k < nBins; k++) {
      final f = k * fs / n;
      if (f >= 0.8 && f <= 5.0 && (best == -1 || spec[k] > spec[best])) best = k;
    }
    if (best == -1) return [0.0, 0.0];
    return [best * fs / n, spec[best] / total];
  }

  // ------------------------------------------------------------ features

  /// Returns the feature map, or null if either wrist lacks enough samples.
  Map<String, double>? _features() {
    final mags = <String, List<double>>{};
    final f = <String, double>{};

    for (final tag in ['L', 'R']) {
      final buf = _buffers[tag]!;
      if (buf.length < 8) return null;

      final cols = <String, List<double>>{
        'AccX': buf.map((s) => s.ax).toList(),
        'AccY': buf.map((s) => s.ay).toList(),
        'AccZ': buf.map((s) => s.az).toList(),
        'GyroX': buf.map((s) => s.gx).toList(),
        'GyroY': buf.map((s) => s.gy).toList(),
        'GyroZ': buf.map((s) => s.gz).toList(),
      };
      cols.forEach((ax, v) {
        f['${tag}_${ax}_mean'] = _mean(v);
        f['${tag}_${ax}_std'] = _std(v);
      });

      final acc = List<double>.generate(
          buf.length,
          (i) => math.sqrt(buf[i].ax * buf[i].ax +
              buf[i].ay * buf[i].ay +
              buf[i].az * buf[i].az));
      final gyro = List<double>.generate(
          buf.length,
          (i) => math.sqrt(buf[i].gx * buf[i].gx +
              buf[i].gy * buf[i].gy +
              buf[i].gz * buf[i].gz));

      f['${tag}_accmag_mean'] = _mean(acc);
      f['${tag}_accmag_std'] = _std(acc);
      f['${tag}_accmag_range'] =
          acc.reduce(math.max) - acc.reduce(math.min);
      f['${tag}_gyromag_mean'] = _mean(gyro);
      f['${tag}_gyromag_std'] = _std(gyro);

      final dfp = _dominantFreq(acc, buf.map((s) => s.tMs).toList());
      f['${tag}_dom_freq'] = dfp[0];
      f['${tag}_peak_power'] = dfp[1];

      mags[tag] = acc;
    }

    f['xdev_accmag_corr'] = _corr(mags['L']!, mags['R']!);
    return f;
  }

  // ------------------------------------------------------------ forest

  List<double> _treeProba(Map<String, dynamic> t, List<double> x) {
    final left = List<int>.from(t['left']);
    final right = List<int>.from(t['right']);
    final feat = List<int>.from(t['feature']);
    final thr = List<double>.from(
        (t['threshold'] as List).map((v) => (v as num).toDouble()));
    int node = 0;
    while (left[node] != -1) {
      node = (x[feat[node]] <= thr[node]) ? left[node] : right[node];
    }
    return List<double>.from(
        (t['value'][node] as List).map((v) => (v as num).toDouble()));
  }

  /// Current best-guess label, or null if not enough data buffered yet.
  String? predict() {
    final conf = predictWithConfidence();
    return conf?.$1;
  }

  /// (label, probability) or null.
  (String, double)? predictWithConfidence() {
    final fmap = _features();
    if (fmap == null) return null;

    final x = List<double>.filled(featureNames.length, 0.0);
    for (int i = 0; i < featureNames.length; i++) {
      final v = fmap[featureNames[i]];
      if (v == null) return null; // feature-set drift guard
      x[i] = v.isFinite ? v : 0.0;
    }

    final probs = List<double>.filled(classes.length, 0.0);
    for (final t in trees) {
      final p = _treeProba(t, x);
      for (int c = 0; c < classes.length; c++) probs[c] += p[c];
    }
    int best = 0;
    for (int c = 1; c < classes.length; c++) {
      if (probs[c] > probs[best]) best = c;
    }
    return (classes[best], probs[best] / trees.length);
  }
}
