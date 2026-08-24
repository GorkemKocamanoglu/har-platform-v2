import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'har_classifier.dart';

// ============================================================ theme
class AppColors {
  static const bg        = Color(0xFF131019); // near-black aubergine
  static const surface   = Color(0xFF1C1726);
  static const surface2  = Color(0xFF251D33);
  static const line      = Color(0xFF33294A);
  static const purple    = Color(0xFF8B5CF6); // identity
  static const purpleBr  = Color(0xFFA78BFA);
  static const purpleDp  = Color(0xFF6D28D9);
  static const green     = Color(0xFF34D399); // live / go
  static const greenBr   = Color(0xFF6EE7B7);
  static const amber     = Color(0xFFFBBF24);
  static const red       = Color(0xFFF87171);
  static const text      = Color(0xFFF4F1FB);
  static const textDim   = Color(0xFFA89FC0);
  static const textFaint = Color(0xFF6F6689);
}

void main() {
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: AppColors.bg,
      colorScheme: const ColorScheme.dark(
        primary: AppColors.purple,
        secondary: AppColors.green,
        surface: AppColors.surface,
      ),
      fontFamily: 'sans-serif',
    ),
    home: const HARDataLogger(),
  ));
}

class HARDataLogger extends StatefulWidget {
  const HARDataLogger({super.key});

  @override
  State<HARDataLogger> createState() => _HARDataLoggerState();
}

class _HARDataLoggerState extends State<HARDataLogger> with SingleTickerProviderStateMixin {
  late TabController _tabController;

  Map<String, BluetoothDevice> connectedDevices = {};
  Map<String, BluetoothCharacteristic> activeCharacteristics = {};
  Map<String, StreamSubscription> _valueSubscriptions = {};
  Map<String, StreamSubscription> _connectionSubscriptions = {};

  // --- per-device health monitoring ---
  Map<String, int> packetCounts = {};      // total packets this session
  Map<String, int> lastPacketMs = {};      // wall-clock ms of last packet
  Map<String, int> malformedCounts = {};   // packets that didn't parse to 6 fields
  Timer? _healthTimer;                     // refreshes the status indicators

  // --- live activity prediction ---
  HarClassifier? _classifier;
  String? _liveLabel;          // null until model loaded + enough data buffered
  double _liveConfidence = 0;
  bool _liveModeEnabled = false;   // false = data-collection, true = recognition
  Timer? _predictTimer;

  final List<String> targetDeviceNames = [
    "HARNode_Ankle_Right",
    "HARNode_Ankle_Left",
    "HARNode_Wrist_Right",
    "HARNode_Wrist_Left"
  ];

  String? primaryDeviceForChart;
  bool isScanning = false;
  bool isRecording = false;
  StreamSubscription? _scanSubscription;

  final String SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b";
  final String CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8";

  // The dropdown is now a GROUND-TRUTH HINT stored in session metadata,
  // not the label itself (labels come from the VLM pipeline).
  String selectedLabel = "Walking";
  final List<String> activities = [
    "Walking", "Running", "Standing", "Sitting", "Cycling",
    "Car", "Bus", "Train", "Scooter", "Mixed", "Other"
  ];
  String wearerId = "P01"; // editable in Settings; stored in session.json

  // --- settings (persisted via shared_preferences) ---
  bool _showActivityHint = false;   // hide the ground-truth hint dropdown by default
  bool _liveModeDefault  = false;   // live recognition off by default
  int? _defaultDuration  = null;    // preferred duration (null=manual, 3, 5)

  // --- streaming write state ---
  IOSink? _csvSink;
  List<String> _writeBuffer = [];
  Timer? _flushTimer;
  Directory? _currentSessionDir;
  String? _currentSessionId;
  int? _recordingStartMs;

  // --- recording duration ---
  int? _recordingLimitMinutes;   // null = unlimited (manual stop only)
  Timer? _autoStopTimer;
  final List<int?> _durationOptions = [null, 3, 5]; // null = manual

  // --- chart (throttled) ---
  List<FlSpot> dataX = [];
  List<FlSpot> dataY = [];
  List<FlSpot> dataZ = [];
  final List<List<double>> _chartBuffer = []; // [ax, ay, az] pending points
  Timer? _chartTimer;
  double timeCounter = 0;

  List<Directory> savedSessions = [];

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _loadSettings();
    _tabController.addListener(_loadSavedSessions);

    FlutterBluePlus.adapterState.listen((state) {
      if (state == BluetoothAdapterState.on) {
        startScan();
      }
    });

    // load the on-device activity model (assets/har_model.json)
    HarClassifier.loadFromAsset('assets/har_model.json').then((c) {
      if (mounted) setState(() => _classifier = c);
    }).catchError((e) {
      debugPrint("Could not load har_model.json: $e");
    });

    // predict once per second from the rolling sensor buffers (only in live mode)
    _predictTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted || !_liveModeEnabled) return;
      final r = _classifier?.predictWithConfidence();
      setState(() {
        _liveLabel = r?.$1;
        _liveConfidence = r?.$2 ?? 0;
      });
    });

    // repaint chart at 10 Hz instead of on every BLE packet
    _chartTimer = Timer.periodic(const Duration(milliseconds: 100), (_) => _drainChartBuffer());
    // refresh device-health indicators at 1 Hz
    _healthTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() {});
      _warnIfNodeStale();
    });

    _loadSavedSessions();
  }

  // ============================================================ settings

  Future<void> _loadSettings() async {
    final p = await SharedPreferences.getInstance();
    if (!mounted) return;
    setState(() {
      wearerId          = p.getString('wearerId') ?? wearerId;
      _showActivityHint = p.getBool('showActivityHint') ?? false;
      _liveModeDefault  = p.getBool('liveModeDefault') ?? false;
      final d = p.getInt('defaultDuration'); // -1 sentinel = manual(null)
      _defaultDuration  = (d == null || d == -1) ? null : d;
      // apply defaults to the live session
      _liveModeEnabled     = _liveModeDefault;
      _recordingLimitMinutes = _defaultDuration;
    });
  }

  Future<void> _saveSetting(String key, Object? value) async {
    final p = await SharedPreferences.getInstance();
    if (value == null) {
      await p.setInt(key, -1);              // sentinel for "manual/none"
    } else if (value is bool) {
      await p.setBool(key, value);
    } else if (value is int) {
      await p.setInt(key, value);
    } else if (value is String) {
      await p.setString(key, value);
    }
  }

  void _openSettings() {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => _SettingsScreen(
        wearerId: wearerId,
        showActivityHint: _showActivityHint,
        liveModeDefault: _liveModeDefault,
        defaultDuration: _defaultDuration,
        modelLoaded: _classifier != null,
        modelClasses: _classifier?.classes,
        onChanged: (wid, hint, liveDef, dur) {
          setState(() {
            wearerId = wid;
            _showActivityHint = hint;
            _liveModeDefault = liveDef;
            _defaultDuration = dur;
            if (!isRecording) _recordingLimitMinutes = dur;
          });
          _saveSetting('wearerId', wid);
          _saveSetting('showActivityHint', hint);
          _saveSetting('liveModeDefault', liveDef);
          _saveSetting('defaultDuration', dur);
        },
        onDeleteAll: _deleteAllSessions,
      ),
    ));
  }

  Future<void> _deleteAllSessions() async {
    final docs = await getApplicationDocumentsDirectory();
    final root = Directory('${docs.path}/sessions');
    if (await root.exists()) await root.delete(recursive: true);
    _loadSavedSessions();
  }

  @override
  void dispose() {
    _chartTimer?.cancel();
    _healthTimer?.cancel();
    _predictTimer?.cancel();
    _flushTimer?.cancel();
    _autoStopTimer?.cancel();
    _scanSubscription?.cancel();
    for (final s in _valueSubscriptions.values) { s.cancel(); }
    for (final s in _connectionSubscriptions.values) { s.cancel(); }
    _csvSink?.close();
    super.dispose();
  }

  // ============================================================ BLE

  void startScan() async {
    setState(() => isScanning = true);

    // single scan subscription — repeated startScan calls must not stack listeners
    _scanSubscription?.cancel();
    _scanSubscription = FlutterBluePlus.scanResults.listen((results) {
      for (ScanResult r in results) {
        String deviceName = r.device.platformName;
        if (targetDeviceNames.contains(deviceName) && !connectedDevices.containsKey(deviceName)) {
          setState(() {
            connectedDevices[deviceName] = r.device;
            primaryDeviceForChart ??= deviceName;
          });
          connectToDevice(r.device, deviceName);
        }
      }
    });

    await FlutterBluePlus.startScan(timeout: const Duration(seconds: 10));
    Future.delayed(const Duration(seconds: 10), () {
      if (mounted) setState(() => isScanning = false);
    });
  }

  void connectToDevice(BluetoothDevice device, String deviceName) async {
    try {
      await device.connect(license: License.nonprofit);

      // watch for disconnects — the silent killer of recordings
      _connectionSubscriptions[deviceName]?.cancel();
      _connectionSubscriptions[deviceName] =
          device.connectionState.listen((BluetoothConnectionState state) {
        if (state == BluetoothConnectionState.disconnected) {
          if (mounted) {
            setState(() {}); // health indicator turns red via lastPacketMs staleness
            if (isRecording) {
              ScaffoldMessenger.of(context).showSnackBar(SnackBar(
                content: Text("WARNING: $deviceName DISCONNECTED during recording!"),
                backgroundColor: AppColors.red,
                duration: const Duration(seconds: 5),
              ));
            }
          }
          // modest auto-reconnect attempt
          Future.delayed(const Duration(seconds: 2), () async {
            try {
              await device.connect(license: License.nonprofit);
              discoverServices(device, deviceName);
            } catch (_) {/* next health tick keeps showing red */}
          });
        }
      });

      discoverServices(device, deviceName);
    } catch (e) {
      debugPrint("Connection Error ($deviceName): $e");
    }
  }

  void discoverServices(BluetoothDevice device, String deviceName) async {
    List<BluetoothService> services = await device.discoverServices();
    for (BluetoothService service in services) {
      if (service.uuid.toString() == SERVICE_UUID) {
        for (BluetoothCharacteristic characteristic in service.characteristics) {
          if (characteristic.uuid.toString() == CHARACTERISTIC_UUID) {
            activeCharacteristics[deviceName] = characteristic;
            await characteristic.setNotifyValue(true);

            // cancel any previous listener for this device (reconnects!)
            _valueSubscriptions[deviceName]?.cancel();
            _valueSubscriptions[deviceName] =
                characteristic.lastValueStream.listen((value) {
              if (value.isEmpty) return;
              String decodedData = utf8.decode(value, allowMalformed: true).trim();
              List<String> parts = decodedData.split(',');

              final nowMs = DateTime.now().millisecondsSinceEpoch;
              if (parts.length == 6) {
                packetCounts[deviceName] = (packetCounts[deviceName] ?? 0) + 1;
                lastPacketMs[deviceName] = nowMs;

                if (isRecording) {
                  _writeBuffer.add("$nowMs,$deviceName,$decodedData");
                }

                // parse all 6 axes once — reused by chart and classifier
                final ax = double.tryParse(parts[0]) ?? 0.0;
                final ay = double.tryParse(parts[1]) ?? 0.0;
                final az = double.tryParse(parts[2]) ?? 0.0;
                final gx = double.tryParse(parts[3]) ?? 0.0;
                final gy = double.tryParse(parts[4]) ?? 0.0;
                final gz = double.tryParse(parts[5]) ?? 0.0;

                // feed the live classifier (rolling 2s buffers per wrist)
                if (_liveModeEnabled) {
                  _classifier?.addSample(deviceName, nowMs, ax, ay, az, gx, gy, gz);
                }

                if (deviceName == primaryDeviceForChart) {
                  _chartBuffer.add([ax, ay, az]); // no setState here — timer drains it
                }
              } else {
                malformedCounts[deviceName] = (malformedCounts[deviceName] ?? 0) + 1;
              }
            });
          }
        }
      }
    }
  }

  void _warnIfNodeStale() {
    if (!isRecording || !mounted) return;
    final now = DateTime.now().millisecondsSinceEpoch;
    for (final name in connectedDevices.keys) {
      final last = lastPacketMs[name];
      if (last != null && now - last > 5000) {
        // stale >5s during recording: surface it once per staleness episode
        // (indicator dot is already red; snackbar only on the transition)
        if (now - last < 6000) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(
            content: Text("WARNING: no data from $name for 5s!"),
            backgroundColor: AppColors.amber,
          ));
        }
      }
    }
  }

  // ============================================================ chart

  // Switch which device feeds the live chart. Clears the plot so the
  // new sensor's trace starts fresh rather than continuing the old line.
  void _selectChartDevice(String name) {
    if (name == primaryDeviceForChart) return;
    setState(() {
      primaryDeviceForChart = name;
      _chartBuffer.clear();
      dataX.clear();
      dataY.clear();
      dataZ.clear();
      timeCounter = 0;
    });
  }

  void _drainChartBuffer() {
    if (_chartBuffer.isEmpty || !mounted) return;
    setState(() {
      for (final p in _chartBuffer) {
        timeCounter += 0.02;
        dataX.add(FlSpot(timeCounter, p[0]));
        dataY.add(FlSpot(timeCounter, p[1]));
        dataZ.add(FlSpot(timeCounter, p[2]));
      }
      _chartBuffer.clear();
      while (dataX.length > 100) {
        dataX.removeAt(0);
        dataY.removeAt(0);
        dataZ.removeAt(0);
      }
    });
  }

  // ============================================================ recording

  String _makeSessionId() {
    final n = DateTime.now();
    String p(int v) => v.toString().padLeft(2, '0');
    return "${n.year}-${p(n.month)}-${p(n.day)}_${p(n.hour)}${p(n.minute)}${p(n.second)}"
        "_${selectedLabel.toLowerCase().replaceAll(' ', '')}";
  }

  String _positionFromName(String deviceName) {
    // "HARNode_Wrist_Left" -> "wrist_left"
    return deviceName.replaceFirst("HARNode_", "").toLowerCase();
  }

  Future<void> startRecording() async {
    final docs = await getApplicationDocumentsDirectory();
    _currentSessionId = _makeSessionId();
    _currentSessionDir = Directory('${docs.path}/sessions/$_currentSessionId');
    await _currentSessionDir!.create(recursive: true);

    final csvFile = File('${_currentSessionDir!.path}/sensors.csv');
    _csvSink = csvFile.openWrite(mode: FileMode.write);
    _csvSink!.writeln("Timestamp_ms,Device_Name,AccX,AccY,AccZ,GyroX,GyroY,GyroZ");

    packetCounts.clear();
    malformedCounts.clear();
    _writeBuffer.clear();
    _recordingStartMs = DateTime.now().millisecondsSinceEpoch;

    // flush buffered lines to disk every 2 seconds — a crash loses at most 2s
    _flushTimer = Timer.periodic(const Duration(seconds: 2), (_) => _flushBuffer());

    setState(() => isRecording = true);

    // auto-stop after the selected limit (manual stop still works as early exit)
    _autoStopTimer?.cancel();
    if (_recordingLimitMinutes != null) {
      _autoStopTimer = Timer(Duration(minutes: _recordingLimitMinutes!), () {
        if (isRecording) stopRecording();
      });
    }
  }

  void _flushBuffer() {
    if (_csvSink == null || _writeBuffer.isEmpty) return;
    final lines = List<String>.from(_writeBuffer);
    _writeBuffer.clear();
    for (final l in lines) {
      _csvSink!.writeln(l);
    }
  }

  Future<void> stopRecording() async {
    setState(() => isRecording = false);
    _flushTimer?.cancel();
    _autoStopTimer?.cancel();
    _flushBuffer();
    await _csvSink?.flush();
    await _csvSink?.close();
    _csvSink = null;

    final stopMs = DateTime.now().millisecondsSinceEpoch;

    // ---- session.json: device identifiers, body positions, timing, ground-truth hint
    final meta = {
      "session_id": _currentSessionId,
      "wearer_id": wearerId,
      "recording_start_ms": _recordingStartMs,
      "recording_stop_ms": stopMs,
      "duration_s": ((stopMs - (_recordingStartMs ?? stopMs)) / 1000).round(),
      "duration_limit_minutes": _recordingLimitMinutes,   // null = manual stop
      if (_showActivityHint) "ground_truth_hint": selectedLabel,
      "video_source": "meta_glasses_separate", // video recorded on glasses, pulled manually
      "timestamp_source": "phone_clock_at_ble_arrival",
      "devices": connectedDevices.keys.map((name) => {
            "name": name,
            "position": _positionFromName(name),
            "packets_received": packetCounts[name] ?? 0,
            "malformed_packets": malformedCounts[name] ?? 0,
          }).toList(),
    };
    final metaFile = File('${_currentSessionDir!.path}/session.json');
    await metaFile.writeAsString(const JsonEncoder.withIndent("  ").convert(meta));

    // ---- post-recording sanity check: did every connected node actually log?
    final silent = connectedDevices.keys
        .where((n) => (packetCounts[n] ?? 0) == 0)
        .toList();
    if (mounted) {
      if (silent.isNotEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text("Saved, BUT NO DATA from: ${silent.join(', ')}"),
          backgroundColor: AppColors.red,
          duration: const Duration(seconds: 8),
        ));
      } else {
        final counts = connectedDevices.keys
            .map((n) => "$n: ${packetCounts[n]}")
            .join(", ");
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text("Saved $_currentSessionId  ($counts)"),
          backgroundColor: AppColors.green,
        ));
      }
    }
    _loadSavedSessions();
  }

  void toggleRecording() async {
    if (isRecording) {
      await stopRecording();
    } else {
      await startRecording();
    }
  }

  // ============================================================ history

  Future<void> _loadSavedSessions() async {
    final docs = await getApplicationDocumentsDirectory();
    final sessionsRoot = Directory('${docs.path}/sessions');
    if (!await sessionsRoot.exists()) {
      setState(() => savedSessions = []);
      return;
    }
    final dirs = sessionsRoot.listSync().whereType<Directory>().toList();
    dirs.sort((a, b) => b.path.compareTo(a.path)); // ids sort chronologically
    setState(() => savedSessions = dirs);
  }

  Future<void> _shareSession(Directory dir) async {
    final files = dir
        .listSync()
        .whereType<File>()
        .map((f) => XFile(f.path))
        .toList();
    if (files.isNotEmpty) {
      await Share.shareXFiles(files, text: 'HAR session: ${dir.path.split('/').last}');
    }
  }

  Future<void> _renameSession(Directory dir) async {
    final current = dir.path.split(Platform.pathSeparator).last;
    final controller = TextEditingController(text: current);
    final newName = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.surface2,
        title: const Text("Rename session", style: TextStyle(color: AppColors.text, fontSize: 16)),
        content: TextField(
          controller: controller,
          autofocus: true,
          style: const TextStyle(color: AppColors.text),
          decoration: const InputDecoration(
            hintText: "New name",
            hintStyle: TextStyle(color: AppColors.textFaint),
            enabledBorder: UnderlineInputBorder(borderSide: BorderSide(color: AppColors.line)),
            focusedBorder: UnderlineInputBorder(borderSide: BorderSide(color: AppColors.purple)),
          ),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text("Cancel", style: TextStyle(color: AppColors.textDim))),
          TextButton(onPressed: () => Navigator.pop(ctx, controller.text.trim()), child: const Text("Rename", style: TextStyle(color: AppColors.purpleBr))),
        ],
      ),
    );
    if (newName == null || newName.isEmpty || newName == current) return;
    // sanitize: no path separators or characters that break folder names
    final safe = newName.replaceAll(RegExp(r'[\\/:*?"<>|]'), '_');
    final parent = dir.parent.path;
    final target = Directory('$parent${Platform.pathSeparator}$safe');
    if (await target.exists()) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text("A session with that name already exists"),
          backgroundColor: AppColors.red));
      }
      return;
    }
    await dir.rename(target.path);
    _loadSavedSessions();
  }

  Future<void> _deleteSession(Directory dir) async {
    final name = dir.path.split('/').last;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.surface2,
        title: const Text("Delete session?"),
        content: Text("Permanently delete '$name'?"),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text("Cancel")),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text("Delete", style: TextStyle(color: AppColors.red)),
          ),
        ],
      ),
    );
    if (confirmed == true) {
      dir.deleteSync(recursive: true);
      _loadSavedSessions();
    }
  }

  // ============================================================ UI helpers

  // short tag for the chart switcher: "HARNode_Wrist_Left" -> "W·L"
  String _shortTag(String name) {
    final n = name.toLowerCase();
    final limb = n.contains("wrist") ? "W" : n.contains("ankle") ? "A" : "?";
    final side = n.contains("left") ? "L" : n.contains("right") ? "R" : "?";
    return "$limb·$side";
  }

  // "HARNode_Wrist_Left" -> limb "Wrist", side "left"
  (String, String) _limbSide(String name) {
    final n = name.toLowerCase();
    final limb = n.contains("wrist") ? "Wrist" : n.contains("ankle") ? "Ankle" : "Node";
    final side = n.contains("left") ? "left" : n.contains("right") ? "right" : "";
    return (limb, side);
  }

  Color _deviceStatusColor(String name) {
    final last = lastPacketMs[name];
    if (last == null) return AppColors.textFaint;
    final age = DateTime.now().millisecondsSinceEpoch - last;
    if (age < 2000) return AppColors.green;
    if (age < 5000) return AppColors.amber;
    return AppColors.red;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _buildAppBar(),
            Expanded(
              child: TabBarView(
                controller: _tabController,
                children: [
                  _buildLiveTab(),
                  _buildHistoryTab(),
                ],
              ),
            ),
            _buildTabBar(),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------- app bar
  Widget _buildAppBar() {
    final connected = connectedDevices.length;
    final anyLive = connectedDevices.keys.any((n) => _deviceStatusColor(n) == AppColors.green);
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 18, 20, 16),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft, end: Alignment.bottomRight,
          colors: [Color(0xFF2A1F42), AppColors.surface],
        ),
        border: Border(bottom: BorderSide(color: AppColors.line)),
      ),
      child: Column(
        children: [
          Row(
            children: [
              // brand mark
              Container(
                width: 38, height: 38,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(11),
                  gradient: const LinearGradient(
                    begin: Alignment.topLeft, end: Alignment.bottomRight,
                    colors: [AppColors.purple, AppColors.purpleDp],
                  ),
                  boxShadow: [BoxShadow(color: AppColors.purple.withOpacity(.45), blurRadius: 14, offset: const Offset(0, 4))],
                ),
                child: const Icon(Icons.show_chart, color: Colors.white, size: 22),
              ),
              const SizedBox(width: 11),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: const [
                  Text("Kinetic", style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, letterSpacing: -.3, color: AppColors.text)),
                  SizedBox(height: 1),
                  Text("HAR · multi-node", style: TextStyle(fontSize: 11, color: AppColors.textFaint, letterSpacing: .5)),
                ],
              ),
              const Spacer(),
              // rescan button
              InkWell(
                onTap: isScanning ? null : startScan,
                borderRadius: BorderRadius.circular(12),
                child: Container(
                  width: 40, height: 40,
                  decoration: BoxDecoration(
                    color: AppColors.surface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppColors.line),
                  ),
                  child: isScanning
                      ? const Padding(
                          padding: EdgeInsets.all(11),
                          child: CircularProgressIndicator(strokeWidth: 2, color: AppColors.purpleBr))
                      : const Icon(Icons.refresh, color: AppColors.textDim, size: 20),
                ),
              ),
              const SizedBox(width: 10),
              // settings button
              InkWell(
                onTap: _openSettings,
                borderRadius: BorderRadius.circular(12),
                child: Container(
                  width: 40, height: 40,
                  decoration: BoxDecoration(
                    color: AppColors.surface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppColors.line),
                  ),
                  child: const Icon(Icons.settings_outlined, color: AppColors.textDim, size: 20),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          // node summary line
          Row(
            children: [
              RichText(
                text: TextSpan(children: [
                  TextSpan(text: "$connected", style: const TextStyle(fontSize: 26, fontWeight: FontWeight.w700, color: AppColors.greenBr, height: 1)),
                  const TextSpan(text: " /4", style: TextStyle(fontSize: 13, color: AppColors.textFaint)),
                ]),
              ),
              const SizedBox(width: 10),
              const Text("nodes\nstreaming", style: TextStyle(fontSize: 12, color: AppColors.textDim, height: 1.2)),
              const Spacer(),
              if (anyLive) _liveTag(),
            ],
          ),
        ],
      ),
    );
  }

  Widget _liveTag() {
    return Row(
      children: [
        Container(
          width: 7, height: 7,
          decoration: BoxDecoration(
            color: AppColors.green, shape: BoxShape.circle,
            boxShadow: [BoxShadow(color: AppColors.green.withOpacity(.6), blurRadius: 6, spreadRadius: 1)],
          ),
        ),
        const SizedBox(width: 6),
        const Text("LIVE", style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, letterSpacing: 1, color: AppColors.green)),
      ],
    );
  }

  // ---------------------------------------------------------- live tab
  Widget _buildLiveTab() {
    return Column(
      children: [
        Expanded(
          child: SingleChildScrollView(
            padding: const EdgeInsets.fromLTRB(20, 18, 20, 18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildLivePredictionPanel(),
                const SizedBox(height: 22),
                _sectionLabel("Sensor nodes"),
                const SizedBox(height: 12),
                _buildNodeGrid(),
                const SizedBox(height: 22),
                _sectionLabel("Live signal"),
                const SizedBox(height: 12),
                _buildChartCard(),
              ],
            ),
          ),
        ),
        _buildControls(),
      ],
    );
  }

  // ---------------------------------------------------------- live prediction
  Widget _buildLivePredictionPanel() {
    final on = _liveModeEnabled;
    final hasModel = _classifier != null;
    final label = !hasModel
        ? "model not loaded"
        : (_liveLabel ?? "waiting for data…");
    final pct = (_liveConfidence * 100).round();

    return Container(
      decoration: BoxDecoration(
        gradient: on
            ? const LinearGradient(
                colors: [AppColors.purpleDp, AppColors.surface2],
                begin: Alignment.topLeft, end: Alignment.bottomRight)
            : null,
        color: on ? null : AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: on ? AppColors.purple : AppColors.line),
      ),
      padding: const EdgeInsets.fromLTRB(18, 14, 12, 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(on ? Icons.sensors : Icons.sensors_off,
                  size: 16, color: on ? AppColors.greenBr : AppColors.textFaint),
              const SizedBox(width: 8),
              Text(
                on ? "LIVE ACTIVITY" : "ACTIVITY RECOGNITION",
                style: TextStyle(
                    fontSize: 11, fontWeight: FontWeight.w600, letterSpacing: 1.4,
                    color: on ? AppColors.greenBr : AppColors.textFaint),
              ),
              const Spacer(),
              Switch(
                value: on,
                activeColor: AppColors.greenBr,
                onChanged: isRecording
                    ? null
                    : (v) => setState(() {
                          _liveModeEnabled = v;
                          _liveLabel = null;
                          _liveConfidence = 0;
                        }),
              ),
            ],
          ),
          if (on) ...[
            const SizedBox(height: 4),
            Row(
              crossAxisAlignment: CrossAxisAlignment.baseline,
              textBaseline: TextBaseline.alphabetic,
              children: [
                Flexible(
                  child: Text(
                    label,
                    style: const TextStyle(
                        fontSize: 30, fontWeight: FontWeight.w700, color: AppColors.text),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                if (_liveLabel != null) ...[
                  const SizedBox(width: 10),
                  Text("$pct%",
                      style: const TextStyle(
                          fontSize: 15, fontWeight: FontWeight.w600, color: AppColors.greenBr)),
                ],
              ],
            ),
          ] else
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                "Off — data-collection mode. Turn on to predict from live sensors.",
                style: const TextStyle(fontSize: 12, color: AppColors.textFaint),
              ),
            ),
        ],
      ),
    );
  }

  Widget _sectionLabel(String text) {
    return Row(
      children: [
        Text(text.toUpperCase(), style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, letterSpacing: 1.4, color: AppColors.textFaint)),
        const SizedBox(width: 8),
        const Expanded(child: Divider(color: AppColors.line, height: 1)),
      ],
    );
  }

  // ---------------------------------------------------------- node grid
  Widget _buildNodeGrid() {
    if (connectedDevices.isEmpty) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.symmetric(vertical: 34),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.line),
        ),
        child: Column(
          children: [
            Icon(isScanning ? Icons.bluetooth_searching : Icons.bluetooth_disabled, color: AppColors.textFaint, size: 30),
            const SizedBox(height: 10),
            Text(isScanning ? "Scanning for nodes…" : "No nodes connected",
                style: const TextStyle(color: AppColors.textDim, fontSize: 13, fontWeight: FontWeight.w600)),
            const SizedBox(height: 4),
            const Text("Tap refresh to scan again", style: TextStyle(color: AppColors.textFaint, fontSize: 11)),
          ],
        ),
      );
    }
    final names = connectedDevices.keys.toList();
    return GridView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: names.length,
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2, mainAxisSpacing: 10, crossAxisSpacing: 10, childAspectRatio: 1.55,
      ),
      itemBuilder: (_, i) => _buildNodeCard(names[i]),
    );
  }

  Widget _buildNodeCard(String name) {
    final color = _deviceStatusColor(name);
    final isLive = color == AppColors.green;
    final (limb, side) = _limbSide(name);
    final count = packetCounts[name] ?? 0;
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: isLive ? AppColors.green.withOpacity(.35) : AppColors.line),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(width: 3, decoration: BoxDecoration(
            color: color == AppColors.textFaint ? Colors.transparent : color,
            borderRadius: const BorderRadius.only(topLeft: Radius.circular(14), bottomLeft: Radius.circular(14)),
          )),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(13, 12, 13, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Container(width: 8, height: 8, decoration: BoxDecoration(
                        color: color, shape: BoxShape.circle,
                        boxShadow: color == AppColors.textFaint ? null : [BoxShadow(color: color, blurRadius: 8)],
                      )),
                      const SizedBox(width: 7),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(limb, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: AppColors.text)),
                          Text(side.toUpperCase(), style: const TextStyle(fontSize: 10, color: AppColors.textFaint, letterSpacing: .5)),
                        ],
                      ),
                    ],
                  ),
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.baseline,
                    textBaseline: TextBaseline.alphabetic,
                    children: [
                      Text("$count", style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w700, color: AppColors.text, letterSpacing: -.5)),
                      const SizedBox(width: 4),
                      const Text("pkts", style: TextStyle(fontSize: 10, color: AppColors.textFaint)),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------- chart card
  Widget _buildChartCard() {
    final names = connectedDevices.keys.toList();
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topCenter, end: Alignment.bottomCenter,
          colors: [AppColors.surface, Color(0xFF191322)],
        ),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.line),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // sensor switcher
          if (names.isNotEmpty)
            Row(
              children: names.map((n) {
                final selected = n == primaryDeviceForChart;
                return Expanded(
                  child: Padding(
                    padding: const EdgeInsets.only(right: 4),
                    child: InkWell(
                      onTap: () => _selectChartDevice(n),
                      borderRadius: BorderRadius.circular(8),
                      child: Container(
                        padding: const EdgeInsets.symmetric(vertical: 6),
                        alignment: Alignment.center,
                        decoration: BoxDecoration(
                          gradient: selected
                              ? const LinearGradient(colors: [AppColors.purple, AppColors.purpleDp])
                              : null,
                          color: selected ? null : AppColors.surface2,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: selected ? Colors.transparent : AppColors.line),
                        ),
                        child: Text(_shortTag(n), style: TextStyle(
                          fontSize: 11, fontWeight: FontWeight.w600, letterSpacing: .5,
                          color: selected ? Colors.white : AppColors.textDim,
                        )),
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
          if (names.isNotEmpty) const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text("Acceleration", style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: AppColors.textDim)),
              Text(primaryDeviceForChart != null ? _positionFromName(primaryDeviceForChart!) : "—",
                  style: const TextStyle(fontSize: 10, color: AppColors.textFaint)),
            ],
          ),
          const SizedBox(height: 8),
          // legend
          Row(
            children: const [
              _LegendItem(color: AppColors.purpleBr, label: "X"),
              SizedBox(width: 12),
              _LegendItem(color: AppColors.green, label: "Y"),
              SizedBox(width: 12),
              _LegendItem(color: AppColors.amber, label: "Z"),
            ],
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 130,
            child: dataX.isEmpty
                ? const Center(child: Text("Waiting for data…", style: TextStyle(color: AppColors.textFaint, fontSize: 12)))
                : LineChart(
                    LineChartData(
                      gridData: FlGridData(show: true, drawVerticalLine: false,
                        getDrawingHorizontalLine: (_) => const FlLine(color: AppColors.line, strokeWidth: .5, dashArray: [3, 4])),
                      titlesData: const FlTitlesData(show: false),
                      borderData: FlBorderData(show: false),
                      minY: -2.0, maxY: 2.0,
                      lineBarsData: [
                        _line(dataX, AppColors.purpleBr),
                        _line(dataY, AppColors.green),
                        _line(dataZ, AppColors.amber),
                      ],
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  LineChartBarData _line(List<FlSpot> spots, Color color) => LineChartBarData(
        spots: spots, isCurved: true, color: color, barWidth: 1.8,
        dotData: const FlDotData(show: false),
      );

  // ---------------------------------------------------------- controls
  Widget _buildControls() {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 20),
      decoration: const BoxDecoration(
        border: Border(top: BorderSide(color: AppColors.line)),
        gradient: LinearGradient(begin: Alignment.bottomCenter, end: Alignment.topCenter,
          colors: [Color(0xFF191322), Colors.transparent]),
      ),
      child: Column(
        children: [
          // hint (optional) + duration
          if (_showActivityHint)
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(child: _hintField()),
                const SizedBox(width: 12),
                Expanded(child: _durationField()),
              ],
            )
          else
            _durationField(),
          const SizedBox(height: 14),
          _buildRecordButton(),
        ],
      ),
    );
  }

  Widget _hintField() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text("ACTIVITY HINT", style: TextStyle(fontSize: 10, color: AppColors.textFaint, letterSpacing: .8, fontWeight: FontWeight.w600)),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
          decoration: BoxDecoration(
            color: AppColors.surface2,
            borderRadius: BorderRadius.circular(11),
            border: Border.all(color: AppColors.line),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: selectedLabel,
              isExpanded: true,
              dropdownColor: AppColors.surface2,
              icon: const Icon(Icons.keyboard_arrow_down, color: AppColors.textFaint, size: 18),
              style: const TextStyle(fontSize: 13, color: AppColors.text, fontWeight: FontWeight.w500),
              items: activities.map((a) => DropdownMenuItem(value: a, child: Text(a))).toList(),
              onChanged: isRecording ? null : (v) => setState(() => selectedLabel = v!),
            ),
          ),
        ),
      ],
    );
  }

  Widget _durationField() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text("DURATION", style: TextStyle(fontSize: 10, color: AppColors.textFaint, letterSpacing: .8, fontWeight: FontWeight.w600)),
        const SizedBox(height: 6),
        Container(
          padding: const EdgeInsets.all(3),
          decoration: BoxDecoration(
            color: AppColors.surface2,
            borderRadius: BorderRadius.circular(11),
            border: Border.all(color: AppColors.line),
          ),
          child: Row(
            children: _durationOptions.map((opt) {
              final selected = _recordingLimitMinutes == opt;
              final label = opt == null ? "Manual" : "${opt} min";
              return Expanded(
                child: InkWell(
                  onTap: isRecording ? null : () => setState(() => _recordingLimitMinutes = opt),
                  borderRadius: BorderRadius.circular(8),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 9),
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      gradient: selected ? const LinearGradient(colors: [AppColors.purple, AppColors.purpleDp]) : null,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(label, style: TextStyle(
                      fontSize: 13, fontWeight: FontWeight.w600,
                      color: selected ? Colors.white : AppColors.textDim,
                    )),
                  ),
                ),
              );
            }).toList(),
          ),
        ),
      ],
    );
  }

  Widget _buildRecordButton() {
    final enabled = connectedDevices.isNotEmpty;
    final durText = _recordingLimitMinutes == null ? "" : " · ${_recordingLimitMinutes} min";
    return Opacity(
      opacity: enabled ? 1 : .45,
      child: InkWell(
        onTap: enabled ? toggleRecording : null,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 17),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(16),
            gradient: LinearGradient(
              begin: Alignment.topLeft, end: Alignment.bottomRight,
              colors: isRecording
                  ? const [Color(0xFFFB7185), AppColors.red]
                  : const [AppColors.greenBr, AppColors.green, Color(0xFF10B981)],
            ),
            boxShadow: [BoxShadow(
              color: (isRecording ? AppColors.red : AppColors.green).withOpacity(.35),
              blurRadius: 24, offset: const Offset(0, 8))],
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 15, height: 15,
                decoration: BoxDecoration(
                  color: isRecording ? Colors.white : const Color(0xFF04120B),
                  borderRadius: BorderRadius.circular(isRecording ? 3 : 50),
                ),
              ),
              const SizedBox(width: 11),
              Text(
                isRecording ? "Stop & save session" : "Start recording",
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600, letterSpacing: .3,
                  color: isRecording ? Colors.white : const Color(0xFF04120B)),
              ),
              if (!isRecording && durText.isNotEmpty)
                Text(durText, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w400, color: Color(0xB304120B))),
            ],
          ),
        ),
      ),
    );
  }

  // ---------------------------------------------------------- tab bar
  Widget _buildTabBar() {
    return Container(
      decoration: const BoxDecoration(
        color: AppColors.bg,
        border: Border(top: BorderSide(color: AppColors.line)),
      ),
      child: Row(
        children: [
          _tabItem(0, Icons.show_chart, "Live"),
          _tabItem(1, Icons.folder_outlined, "Sessions"),
        ],
      ),
    );
  }

  Widget _tabItem(int index, IconData icon, String label) {
    final active = _tabController.index == index;
    return Expanded(
      child: InkWell(
        onTap: () => setState(() => _tabController.animateTo(index)),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(icon, size: 20, color: active ? AppColors.purpleBr : AppColors.textFaint),
              const SizedBox(height: 3),
              Text(label, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w600,
                color: active ? AppColors.purpleBr : AppColors.textFaint)),
            ],
          ),
        ),
      ),
    );
  }

  // ---------------------------------------------------------- history tab
  Widget _buildHistoryTab() {
    if (savedSessions.isEmpty) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.folder_open, color: AppColors.textFaint, size: 34),
            SizedBox(height: 12),
            Text("No sessions yet", style: TextStyle(color: AppColors.textDim, fontSize: 14, fontWeight: FontWeight.w600)),
            SizedBox(height: 4),
            Text("Recorded sessions will appear here", style: TextStyle(color: AppColors.textFaint, fontSize: 12)),
          ],
        ),
      );
    }
    return ListView.builder(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
      itemCount: savedSessions.length,
      itemBuilder: (context, index) {
        final dir = savedSessions[index];
        final name = dir.path.split('/').last;
        int totalKb = 0;
        for (final f in dir.listSync().whereType<File>()) {
          totalKb += f.lengthSync();
        }
        return Container(
          margin: const EdgeInsets.only(bottom: 10),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppColors.line),
          ),
          child: ListTile(
            contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
            leading: Container(
              width: 40, height: 40,
              decoration: BoxDecoration(
                color: AppColors.purple.withOpacity(.15),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.insert_chart_outlined, color: AppColors.purpleBr, size: 20),
            ),
            title: Text(name, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13, color: AppColors.text)),
            subtitle: Text("${(totalKb / 1024).toStringAsFixed(1)} KB", style: const TextStyle(color: AppColors.textFaint, fontSize: 11)),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                IconButton(icon: const Icon(Icons.drive_file_rename_outline, color: AppColors.purpleBr, size: 20), onPressed: () => _renameSession(dir)),
                IconButton(icon: const Icon(Icons.ios_share, color: AppColors.green, size: 20), onPressed: () => _shareSession(dir)),
                IconButton(icon: const Icon(Icons.delete_outline, color: AppColors.red, size: 20), onPressed: () => _deleteSession(dir)),
              ],
            ),
          ),
        );
      },
    );
  }
}

// ============================================================ small widgets
class _LegendItem extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendItem({required this.color, required this.label});
  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(width: 14, height: 2, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2))),
        const SizedBox(width: 5),
        Text(label, style: const TextStyle(fontSize: 10, color: AppColors.textDim)),
      ],
    );
  }
}

// ============================================================ settings screen
class _SettingsScreen extends StatefulWidget {
  final String wearerId;
  final bool showActivityHint;
  final bool liveModeDefault;
  final int? defaultDuration;
  final bool modelLoaded;
  final List<String>? modelClasses;
  final void Function(String wearerId, bool showHint, bool liveDefault, int? duration) onChanged;
  final Future<void> Function() onDeleteAll;

  const _SettingsScreen({
    required this.wearerId,
    required this.showActivityHint,
    required this.liveModeDefault,
    required this.defaultDuration,
    required this.modelLoaded,
    required this.modelClasses,
    required this.onChanged,
    required this.onDeleteAll,
  });

  @override
  State<_SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<_SettingsScreen> {
  late TextEditingController _wearerCtrl;
  late bool _showHint;
  late bool _liveDefault;
  late int? _duration;

  @override
  void initState() {
    super.initState();
    _wearerCtrl = TextEditingController(text: widget.wearerId);
    _showHint = widget.showActivityHint;
    _liveDefault = widget.liveModeDefault;
    _duration = widget.defaultDuration;
  }

  @override
  void dispose() {
    _wearerCtrl.dispose();
    super.dispose();
  }

  void _push() => widget.onChanged(
      _wearerCtrl.text.trim().isEmpty ? "P01" : _wearerCtrl.text.trim(),
      _showHint, _liveDefault, _duration);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      appBar: AppBar(
        backgroundColor: AppColors.bg,
        elevation: 0,
        title: const Text("Settings", style: TextStyle(color: AppColors.text, fontSize: 18, fontWeight: FontWeight.w700)),
        iconTheme: const IconThemeData(color: AppColors.textDim),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(18, 8, 18, 24),
        children: [
          _group("Recording"),
          _card([
            _switchRow(
              "Show activity hint",
              "A dropdown to tag what you're doing. Stored in session metadata; does not affect labels.",
              _showHint,
              (v) { setState(() => _showHint = v); _push(); },
            ),
            _divider(),
            _labelRow("Default duration"),
            const SizedBox(height: 8),
            Row(
              children: [null, 3, 5].map((opt) {
                final sel = _duration == opt;
                final label = opt == null ? "Manual" : "$opt min";
                return Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: ChoiceChip(
                    label: Text(label),
                    selected: sel,
                    backgroundColor: AppColors.surface2,
                    selectedColor: AppColors.purpleDp,
                    labelStyle: TextStyle(color: sel ? Colors.white : AppColors.textDim, fontSize: 13),
                    side: const BorderSide(color: AppColors.line),
                    onSelected: (_) { setState(() => _duration = opt); _push(); },
                  ),
                );
              }).toList(),
            ),
          ]),

          _group("Prediction"),
          _card([
            _switchRow(
              "Live recognition on by default",
              "Start each session with live activity prediction turned on.",
              _liveDefault,
              (v) { setState(() => _liveDefault = v); _push(); },
            ),
          ]),

          _group("Identity"),
          _card([
            _labelRow("Wearer ID"),
            const SizedBox(height: 8),
            TextField(
              controller: _wearerCtrl,
              style: const TextStyle(color: AppColors.text, fontSize: 14),
              onChanged: (_) => _push(),
              decoration: InputDecoration(
                isDense: true,
                filled: true,
                fillColor: AppColors.surface2,
                hintText: "e.g. P01",
                hintStyle: const TextStyle(color: AppColors.textFaint),
                enabledBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: const BorderSide(color: AppColors.line)),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: const BorderSide(color: AppColors.purple)),
              ),
            ),
            const SizedBox(height: 6),
            const Text("Recorded into each session's metadata.",
                style: TextStyle(color: AppColors.textFaint, fontSize: 11)),
          ]),

          _group("Model"),
          _card([
            _infoRow("Status", widget.modelLoaded ? "Loaded" : "Not loaded",
                widget.modelLoaded ? AppColors.greenBr : AppColors.red),
            if (widget.modelClasses != null) ...[
              _divider(),
              _labelRow("Classes"),
              const SizedBox(height: 8),
              Wrap(
                spacing: 6, runSpacing: 6,
                children: widget.modelClasses!.map((c) => Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: AppColors.surface2,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: AppColors.line),
                  ),
                  child: Text(c, style: const TextStyle(color: AppColors.textDim, fontSize: 12)),
                )).toList(),
              ),
            ],
          ]),

          _group("Data"),
          _card([
            InkWell(
              onTap: () async {
                final ok = await showDialog<bool>(
                  context: context,
                  builder: (ctx) => AlertDialog(
                    backgroundColor: AppColors.surface2,
                    title: const Text("Delete all sessions?", style: TextStyle(color: AppColors.text, fontSize: 16)),
                    content: const Text("This permanently removes every recorded session on the phone. This cannot be undone.",
                        style: TextStyle(color: AppColors.textDim, fontSize: 13)),
                    actions: [
                      TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text("Cancel", style: TextStyle(color: AppColors.textDim))),
                      TextButton(onPressed: () => Navigator.pop(ctx, true), child: const Text("Delete all", style: TextStyle(color: AppColors.red))),
                    ],
                  ),
                );
                if (ok == true) {
                  await widget.onDeleteAll();
                  if (mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                      content: Text("All sessions deleted"), backgroundColor: AppColors.red));
                  }
                }
              },
              child: Row(
                children: const [
                  Icon(Icons.delete_forever_outlined, color: AppColors.red, size: 20),
                  SizedBox(width: 12),
                  Text("Delete all sessions", style: TextStyle(color: AppColors.red, fontSize: 14, fontWeight: FontWeight.w600)),
                ],
              ),
            ),
          ]),

          const SizedBox(height: 20),
          const Center(child: Text("Kinetic · HAR multi-node", style: TextStyle(color: AppColors.textFaint, fontSize: 11))),
        ],
      ),
    );
  }

  // -- small building blocks --
  Widget _group(String t) => Padding(
    padding: const EdgeInsets.fromLTRB(4, 20, 4, 8),
    child: Text(t.toUpperCase(), style: const TextStyle(fontSize: 11, letterSpacing: 1.4, fontWeight: FontWeight.w600, color: AppColors.textFaint)),
  );

  Widget _card(List<Widget> children) => Container(
    padding: const EdgeInsets.all(16),
    decoration: BoxDecoration(
      color: AppColors.surface,
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: AppColors.line),
    ),
    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: children),
  );

  Widget _divider() => const Padding(
    padding: EdgeInsets.symmetric(vertical: 12),
    child: Divider(color: AppColors.line, height: 1),
  );

  Widget _labelRow(String t) => Text(t, style: const TextStyle(color: AppColors.text, fontSize: 14, fontWeight: FontWeight.w600));

  Widget _switchRow(String title, String sub, bool value, ValueChanged<bool> onChanged) => Row(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Expanded(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: const TextStyle(color: AppColors.text, fontSize: 14, fontWeight: FontWeight.w600)),
            const SizedBox(height: 3),
            Text(sub, style: const TextStyle(color: AppColors.textFaint, fontSize: 11, height: 1.3)),
          ],
        ),
      ),
      Switch(value: value, activeColor: AppColors.purpleBr, onChanged: onChanged),
    ],
  );

  Widget _infoRow(String k, String v, Color vc) => Row(
    mainAxisAlignment: MainAxisAlignment.spaceBetween,
    children: [
      Text(k, style: const TextStyle(color: AppColors.text, fontSize: 14, fontWeight: FontWeight.w600)),
      Text(v, style: TextStyle(color: vc, fontSize: 13, fontWeight: FontWeight.w600)),
    ],
  );
}