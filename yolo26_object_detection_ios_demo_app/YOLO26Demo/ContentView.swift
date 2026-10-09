import SwiftUI
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif
import AVFoundation
import CoreML
import Vision
import QuartzCore
import UniformTypeIdentifiers

extension Color {
    init(platformColor: PlatformColor) {
        #if canImport(UIKit)
        self.init(uiColor: platformColor)
        #elseif canImport(AppKit)
        self.init(nsColor: platformColor)
        #endif
    }
}

// MARK: - Main TabView

struct ContentView: View {
    @StateObject private var detector = Detector(modelName: "yolo26m-krops-v0.1", labels: Detector.kropsLabels)
    @State private var selectedTab = 0

    var body: some View {
        ZStack {
            TabView(selection: $selectedTab) {
                CameraDetectionView(detector: detector, modeTitle: "Palm Crown")
                    .tabItem { Label("Camera", systemImage: "camera") }
                    .tag(0)

                PhotoDetectionView(detector: detector)
                    .tabItem { Label("Photo", systemImage: "photo") }
                    .tag(1)

                VideoDetectionView(detector: detector)
                    .tabItem { Label("Video", systemImage: "video") }
                    .tag(2)
            }

            if !detector.isReady {
                Color.black.ignoresSafeArea()
                VStack(spacing: 16) {
                    ProgressView()
                        .tint(.white)
                        .scaleEffect(1.2)
                    Text("Loading yolo26m-krops-v0.1 model...")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
        }
        .tint(.white)
        .preferredColorScheme(.dark)
    }
}

// MARK: - Detection Settings Component

struct DetectionControlsView: View {
    @ObservedObject var detector: Detector

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: "slider.horizontal.3")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("Threshold:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("\(Int(detector.confThreshold * 100))%")
                    .font(.caption.monospacedDigit().weight(.semibold))
                    .foregroundColor(.white)
                    .frame(minWidth: 32, alignment: .leading)
                Slider(value: $detector.confThreshold, in: 0.05...0.95, step: 0.05)
                    .tint(Color(platformColor: detector.colors[0]))
            }

            HStack(spacing: 8) {
                Image(systemName: "square.stack.3d.up")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("Max Objects:")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text("\(detector.maxDetections)")
                    .font(.caption.monospacedDigit().weight(.semibold))
                    .foregroundColor(.white)

                Spacer()

                Stepper("", value: $detector.maxDetections, in: 1...100)
                    .labelsHidden()
            }
        }
    }
}

// MARK: - Detection Overlay (shared for photo & video)

struct DetectionOverlay: View {
    let detections: [Detection]
    let imageSize: CGSize
    let displaySize: CGSize
    let colors: [PlatformColor]

    var body: some View {
        let transform = aspectFitTransform()

        // Motion trails for tracked objects, drawn underneath the boxes.
        ForEach(detections) { det in
            if det.trail.count >= 2 {
                let colorIdx = det.trackId ?? det.classIndex
                let color = Color(platformColor: colors[colorIdx % colors.count])
                Path { path in
                    let pts = det.trail.map { scaledPoint($0, transform: transform) }
                    path.move(to: pts[0])
                    for p in pts.dropFirst() { path.addLine(to: p) }
                }
                .stroke(color.opacity(0.75),
                        style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
            }
        }

        ForEach(detections) { det in
            let r = scaledRect(det.normRect, transform: transform)
            // Color by trackId when present so each tracked object keeps
            // its own color across frames; otherwise color by class.
            let colorIdx = det.trackId ?? det.classIndex
            let color = Color(platformColor: colors[colorIdx % colors.count])
            let labelText: String = {
                let teText = CameraVC.extractTahunEmisi(from: det.label)
                let teSuffix = teText.isEmpty ? "" : " [\(teText)]"
                if let tid = det.trackId {
                    return "  #\(tid) \(det.label)\(teSuffix) \(Int(det.confidence * 100))%  "
                } else {
                    return "  \(det.label)\(teSuffix) \(Int(det.confidence * 100))%  "
                }
            }()

            RoundedRectangle(cornerRadius: 10)
                .stroke(color, lineWidth: 2.5)
                .background(RoundedRectangle(cornerRadius: 10).fill(color.opacity(0.12)))
                .frame(width: r.width, height: r.height)
                .position(x: r.midX, y: r.midY)

            Text(labelText)
                .font(.system(size: 11, weight: .bold))
                .foregroundColor(.white)
                .padding(.horizontal, 5)
                .padding(.vertical, 3)
                .background(color)
                .cornerRadius(6)
                .position(x: r.midX, y: r.minY > 20 ? r.minY - 14 : r.maxY + 14)
        }
    }

    private func scaledPoint(_ p: CGPoint, transform t: FitTransform) -> CGPoint {
        CGPoint(x: p.x * imageSize.width * t.scale + t.offsetX,
                y: p.y * imageSize.height * t.scale + t.offsetY)
    }

    private struct FitTransform {
        let scale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat
    }

    private func aspectFitTransform() -> FitTransform {
        guard imageSize.width > 0, imageSize.height > 0 else {
            return FitTransform(scale: 1, offsetX: 0, offsetY: 0)
        }
        let scaleX = displaySize.width / imageSize.width
        let scaleY = displaySize.height / imageSize.height
        let scale = min(scaleX, scaleY)
        let scaledW = imageSize.width * scale
        let scaledH = imageSize.height * scale
        return FitTransform(scale: scale,
                            offsetX: (displaySize.width - scaledW) / 2,
                            offsetY: (displaySize.height - scaledH) / 2)
    }

    private func scaledRect(_ nr: CGRect, transform t: FitTransform) -> CGRect {
        let x = nr.minX * imageSize.width * t.scale + t.offsetX
        let y = nr.minY * imageSize.height * t.scale + t.offsetY
        let w = nr.width * imageSize.width * t.scale
        let h = nr.height * imageSize.height * t.scale
        return CGRect(x: x, y: y, width: w, height: h)
    }
}

// MARK: - Photo Detection

struct PhotoDetectionView: View {
    @ObservedObject var detector: Detector
    @State private var image: PlatformImage?
    @State private var detections: [Detection] = []
    @State private var isProcessing = false
    @State private var inferenceTime: Double = 0
    @State private var showFileImporter = false
    @State private var isDropTargeted = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let image {
                GeometryReader { geo in
                    #if canImport(UIKit)
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    #elseif canImport(AppKit)
                    Image(nsImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    #endif

                    DetectionOverlay(detections: detections,
                                     imageSize: image.size,
                                     displaySize: geo.size,
                                     colors: detector.colors)
                }
                .contentShape(Rectangle())
                .onTapGesture(count: 2) {
                    selectPhotoFromFinder()
                }
            } else {
                Button {
                    selectPhotoFromFinder()
                } label: {
                    VStack(spacing: 14) {
                        Image(systemName: "folder.badge.plus")
                            .font(.system(size: 52))
                        Text("Tap to select photo from Finder")
                            .font(.headline)
                        Text("Click or tap to choose an image, or drag & drop file here")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .foregroundColor(.white.opacity(0.85))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }

            // Top model info badge
            VStack {
                HStack(spacing: 6) {
                    Image(systemName: "camera.macro")
                        .font(.caption)
                    Text("yolo26m-krops-v0.1 • Palm Crown")
                        .font(.caption.weight(.semibold))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(.top, 50)
                Spacer()
            }

            // Bottom bar
            VStack {
                Spacer()
                VStack(spacing: 8) {
                    DetectionControlsView(detector: detector)

                    HStack {
                        if !detections.isEmpty {
                            Text("\(detections.count) palm crowns")
                                .font(.caption)
                        }
                        Spacer()
                        if inferenceTime > 0 {
                            Text(String(format: "%.0fms", inferenceTime))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Button {
                            selectPhotoFromFinder()
                        } label: {
                            Image(systemName: "folder.badge.plus")
                                .font(.body)
                                .foregroundColor(.white)
                        }
                        .buttonStyle(.plain)
                        .help("Choose photo from Finder")
                    }
                }
                .foregroundColor(.white)
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
            }

            if isProcessing {
                ProgressView()
                    .tint(.white)
                    .scaleEffect(1.5)
                    .padding(24)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
        }
        .onDrop(of: [.fileURL, .image], isTargeted: $isDropTargeted) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url = url {
                    DispatchQueue.main.async {
                        loadImage(from: url)
                    }
                }
            }
            return true
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.image],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                loadImage(from: url)
            }
        }
        .onChange(of: detector.confThreshold) { _ in
            if let image {
                runDetection(on: image)
            }
        }
        .onChange(of: detector.maxDetections) { _ in
            if let image {
                runDetection(on: image)
            }
        }
    }

    private func selectPhotoFromFinder() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.prompt = "Select"
        panel.message = "Choose an image for Palm Crown detection"

        NSApp.activate(ignoringOtherApps: true)
        if let window = NSApp.keyWindow ?? NSApp.windows.first(where: { $0.isVisible }) {
            panel.beginSheetModal(for: window) { response in
                if response == .OK, let url = panel.url {
                    loadImage(from: url)
                }
            }
        } else {
            panel.begin { response in
                if response == .OK, let url = panel.url {
                    loadImage(from: url)
                }
            }
        }
        #else
        showFileImporter = true
        #endif
    }

    private func loadImage(from url: URL) {
        let gotAccess = url.startAccessingSecurityScopedResource()
        defer { if gotAccess { url.stopAccessingSecurityScopedResource() } }

        guard let data = try? Data(contentsOf: url),
              let loadedImage = PlatformImage(data: data) else { return }

        DispatchQueue.main.async {
            self.image = loadedImage
            self.runDetection(on: loadedImage)
        }
    }

    private func runDetection(on loadedImage: PlatformImage) {
        isProcessing = true
        Task {
            let start = CFAbsoluteTimeGetCurrent()
            let dets = await detector.detect(image: loadedImage)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            await MainActor.run {
                detections = dets
                inferenceTime = elapsed
                isProcessing = false
            }
        }
    }
}

// MARK: - Video Detection

struct VideoDetectionView: View {
    @ObservedObject var detector: Detector
    @State private var currentFrame: PlatformImage?
    @State private var detections: [Detection] = []
    @State private var isPlaying = false
    @State private var progress: Double = 0
    @State private var fps: Double = 0
    @State private var playbackTask: Task<Void, Never>?
    @State private var trackingEnabled: Bool = true
    @State private var showFileImporter = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let currentFrame {
                GeometryReader { geo in
                    #if canImport(UIKit)
                    Image(uiImage: currentFrame)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    #elseif canImport(AppKit)
                    Image(nsImage: currentFrame)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    #endif

                    DetectionOverlay(detections: detections,
                                     imageSize: currentFrame.size,
                                     displaySize: geo.size,
                                     colors: detector.colors)
                }
                .contentShape(Rectangle())
                .onTapGesture(count: 2) {
                    selectVideoFromFinder()
                }
            } else {
                Button {
                    selectVideoFromFinder()
                } label: {
                    VStack(spacing: 14) {
                        Image(systemName: "film")
                            .font(.system(size: 52))
                        Text("Tap to select video from Finder")
                            .font(.headline)
                        Text("Click or tap to choose a video, or drag & drop file here")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    .foregroundColor(.white.opacity(0.85))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }

            // Top model info badge
            VStack {
                HStack(spacing: 6) {
                    Image(systemName: "camera.macro")
                        .font(.caption)
                    Text("yolo26m-krops-v0.1 • Palm Crown")
                        .font(.caption.weight(.semibold))
                }
                .foregroundColor(.white)
                .padding(.horizontal, 14)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(.top, 50)
                Spacer()
            }

            // Bottom bar
            VStack {
                Spacer()
                VStack(spacing: 8) {
                    if currentFrame != nil {
                        ProgressView(value: progress)
                            .tint(Color(platformColor: detector.colors[0]))
                    }
                    DetectionControlsView(detector: detector)
                    HStack(spacing: 12) {
                        if !detections.isEmpty {
                            Text("\(detections.count) palm crowns")
                                .font(.caption)
                        }
                        Spacer()
                        Button {
                            trackingEnabled.toggle()
                        } label: {
                            HStack(spacing: 4) {
                                Image(systemName: trackingEnabled ? "scope" : "circle.dashed")
                                Text(trackingEnabled ? "Track" : "Raw")
                            }
                            .font(.caption.weight(.semibold))
                            .foregroundColor(trackingEnabled ? .black : .white)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(trackingEnabled ? Color.white : Color.white.opacity(0.15))
                            .cornerRadius(6)
                        }
                        if fps > 0 {
                            Text(String(format: "%.1f FPS", fps))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Button {
                            selectVideoFromFinder()
                        } label: {
                            Image(systemName: "folder.badge.plus")
                                .font(.body)
                                .foregroundColor(.white)
                        }
                        .buttonStyle(.plain)
                        .help("Choose video from Finder")
                    }
                    .foregroundColor(.white)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(.ultraThinMaterial)
            }
        }
        .onDrop(of: [.fileURL, .movie], isTargeted: nil) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url = url {
                    DispatchQueue.main.async {
                        loadAndProcess(url: url)
                    }
                }
            }
            return true
        }
        .fileImporter(
            isPresented: $showFileImporter,
            allowedContentTypes: [.movie, .video, .quickTimeMovie, .mpeg4Movie],
            allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first {
                loadAndProcess(url: url)
            }
        }
        .onDisappear { playbackTask?.cancel() }
    }

    private func selectVideoFromFinder() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .video, .quickTimeMovie, .mpeg4Movie]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.prompt = "Select Video"
        panel.message = "Choose a video for Palm Crown detection & tracking"

        NSApp.activate(ignoringOtherApps: true)
        if let window = NSApp.keyWindow ?? NSApp.windows.first(where: { $0.isVisible }) {
            panel.beginSheetModal(for: window) { response in
                if response == .OK, let url = panel.url {
                    loadAndProcess(url: url)
                }
            }
        } else {
            panel.begin { response in
                if response == .OK, let url = panel.url {
                    loadAndProcess(url: url)
                }
            }
        }
        #else
        showFileImporter = true
        #endif
    }

    private func loadAndProcess(url: URL) {
        playbackTask?.cancel()
        let tracking = trackingEnabled
        let det = detector
        isPlaying = true
        playbackTask = Task.detached(priority: .userInitiated) {
            let gotAccess = url.startAccessingSecurityScopedResource()
            defer { if gotAccess { url.stopAccessingSecurityScopedResource() } }
            await processVideo(url: url, tracking: tracking, detector: det)
        }
    }

    private func processVideo(url: URL, tracking: Bool, detector: Detector) async {
        let asset = AVURLAsset(url: url)
        guard let track = try? await asset.loadTracks(withMediaType: .video).first else { return }
        let duration = try? await asset.load(.duration)
        let totalSeconds = duration.map { CMTimeGetSeconds($0) } ?? 1
        let nominalFPS = (try? await track.load(.nominalFrameRate)) ?? 30
        let frameInterval = 1.0 / Double(nominalFPS)

        guard let reader = try? AVAssetReader(asset: asset) else { return }
        let outputSettings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        let trackOutput = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        reader.add(trackOutput)
        reader.startReading()

        let ciContext = CIContext()
        var frameCount = 0
        let tracker = ByteTracker()

        while !Task.isCancelled, let sb = trackOutput.copyNextSampleBuffer() {
            let pts = CMSampleBufferGetPresentationTimeStamp(sb)
            let currentSec = CMTimeGetSeconds(pts)

            guard let pb = CMSampleBufferGetImageBuffer(sb) else { continue }
            let start = CFAbsoluteTimeGetCurrent()
            let rawDets = detector.detect(pixelBuffer: pb)
            let dets = tracking ? tracker.update(detections: rawDets) : rawDets
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            // Convert pixel buffer to PlatformImage
            let ciImage = CIImage(cvPixelBuffer: pb)
            guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { continue }
            #if canImport(UIKit)
            let frame = UIImage(cgImage: cgImage)
            #elseif canImport(AppKit)
            let frame = NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
            #endif

            frameCount += 1
            let currentFPS = 1.0 / max(elapsed, 0.001)

            await MainActor.run {
                currentFrame = frame
                detections = dets
                progress = currentSec / totalSeconds
                fps = fps == 0 ? currentFPS : fps * 0.9 + currentFPS * 0.1
            }

            let sleepTime = max(frameInterval - elapsed, 0)
            if sleepTime > 0 {
                try? await Task.sleep(for: .seconds(sleepTime))
            }
        }

        await MainActor.run {
            isPlaying = false
            progress = 1.0
        }
    }
}

struct VideoTransferable: Transferable {
    let url: URL
    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString + "." + received.file.pathExtension)
            try FileManager.default.copyItem(at: received.file, to: tmp)
            return Self(url: tmp)
        }
    }
}

// MARK: - Camera Detection (UIKit wrapper)

struct CameraDetectionView: View {
    @ObservedObject var detector: Detector
    var modeTitle: String = "Palm Crown"

    var body: some View {
        ZStack {
            CameraVCWrapper(detector: detector, modeTitle: modeTitle)
                .ignoresSafeArea()

            VStack {
                Spacer()
                DetectionControlsView(detector: detector)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
            }
        }
    }
}

#if canImport(UIKit)
struct CameraVCWrapper: UIViewControllerRepresentable {
    let detector: Detector
    var modeTitle: String = "Cash"

    func makeUIViewController(context: Context) -> CameraVC {
        CameraVC(detector: detector, modeTitle: modeTitle)
    }
    func updateUIViewController(_ vc: CameraVC, context: Context) {}
}

// MARK: - Camera ViewController

class CameraVC: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let detector: Detector
    private let modeTitle: String
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session")
    private let inferenceQueue = DispatchQueue(label: "inference")
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var isProcessing = false

    // Real-time bounding box pool
    private var boxViews: [BoundingBoxView] = []

    // Header badge
    private let modeBadge = UILabel()

    // Big denomination label (updated every ~1 second)
    private let denomLabel = UILabel()
    // Tahun Emisi label below the denomination
    private let tahunEmisiLabel = UILabel()

    private var lastDisplayUpdate: CFAbsoluteTime = 0
    private let displayInterval: CFAbsoluteTime = 1.0

    // Accumulate the best detection across frames between display updates
    private var bestLabel: String = ""
    private var bestConf: Float = 0

    // Stats
    private let statsLabel = CATextLayer()

    init(detector: Detector, modeTitle: String = "Cash") {
        self.detector = detector
        self.modeTitle = modeTitle
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError() }

    override func viewDidLoad() {
        super.viewDidLoad()

        // Camera preview
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        // Add bounding box views to preview layer (drawn behind foreground labels)
        for _ in 0..<30 {
            let bv = BoundingBoxView()
            bv.addToLayer(previewLayer)
            boxViews.append(bv)
        }

        // Header mode badge
        modeBadge.text = "Identifying \(modeTitle) »"
        modeBadge.textAlignment = .center
        modeBadge.font = UIFont.systemFont(ofSize: 14, weight: .semibold)
        modeBadge.textColor = .white
        modeBadge.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        modeBadge.layer.cornerRadius = 14
        modeBadge.clipsToBounds = true
        view.addSubview(modeBadge)

        // Big denomination label
        denomLabel.textAlignment = .center
        denomLabel.numberOfLines = 1
        denomLabel.font = UIFont.systemFont(ofSize: 110, weight: .black)
        denomLabel.textColor = UIColor.systemYellow
        denomLabel.shadowColor = UIColor.black.withAlphaComponent(0.8)
        denomLabel.shadowOffset = CGSize(width: 2, height: 2)
        denomLabel.layer.shadowRadius = 6
        denomLabel.layer.shadowOpacity = 0.8
        denomLabel.adjustsFontSizeToFitWidth = true
        denomLabel.minimumScaleFactor = 0.3
        denomLabel.alpha = 0
        view.addSubview(denomLabel)

        // Tahun Emisi label
        tahunEmisiLabel.textAlignment = .center
        tahunEmisiLabel.font = UIFont.systemFont(ofSize: 18, weight: .bold)
        tahunEmisiLabel.textColor = UIColor.white
        tahunEmisiLabel.shadowColor = UIColor.black.withAlphaComponent(0.8)
        tahunEmisiLabel.shadowOffset = CGSize(width: 1, height: 1)
        tahunEmisiLabel.layer.shadowRadius = 4
        tahunEmisiLabel.layer.shadowOpacity = 0.8
        tahunEmisiLabel.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        tahunEmisiLabel.layer.cornerRadius = 14
        tahunEmisiLabel.clipsToBounds = true
        tahunEmisiLabel.alpha = 0
        view.addSubview(tahunEmisiLabel)

        // Stats overlay
        statsLabel.fontSize = 12
        statsLabel.font = UIFont.monospacedSystemFont(ofSize: 12, weight: .medium)
        statsLabel.foregroundColor = UIColor.white.cgColor
        statsLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5).cgColor
        statsLabel.cornerRadius = 8
        statsLabel.masksToBounds = true
        statsLabel.contentsScale = UIScreen.main.scale
        statsLabel.alignmentMode = .center
        view.layer.addSublayer(statsLabel)

        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard ok else { return }
            self?.sessionQueue.async { self?.setupCamera() }
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds

        modeBadge.frame = CGRect(
            x: (view.bounds.width - 250) / 2,
            y: view.safeAreaInsets.top + 8,
            width: 250,
            height: 28
        )

        statsLabel.frame = CGRect(
            x: (view.bounds.width - 180) / 2,
            y: view.safeAreaInsets.top + 42,
            width: 180,
            height: 22
        )

        let pad: CGFloat = 20
        let denomY = view.bounds.height * 0.28
        let denomH = view.bounds.height * 0.24
        denomLabel.frame = CGRect(
            x: pad,
            y: denomY,
            width: view.bounds.width - pad * 2,
            height: denomH
        )

        tahunEmisiLabel.frame = CGRect(
            x: (view.bounds.width - 240) / 2,
            y: denomY + denomH + 8,
            width: 240,
            height: 32
        )
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        sessionQueue.async { if !self.session.isRunning { self.session.startRunning() } }
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sessionQueue.async { self.session.stopRunning() }
    }

    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .high
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: dev) else { session.commitConfiguration(); return }
        if session.canAddInput(input) { session.addInput(input) }
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: inferenceQueue)
        if session.canAddOutput(out) { session.addOutput(out) }
        session.commitConfiguration()
        out.connection(with: .video)?.videoOrientation = .portrait
        previewLayer.connection?.videoOrientation = .portrait
        session.startRunning()
    }

    // MARK: - Inference (every frame)

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from conn: AVCaptureConnection) {
        guard !isProcessing else { return }
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }

        isProcessing = true
        let start = CACurrentMediaTime()
        let dets = detector.detect(pixelBuffer: pb)
        let ms = (CACurrentMediaTime() - start) * 1000
        isProcessing = false

        // Update real-time bounding boxes and stats on background
        let statsText = String(format: "  %.0f ms  ", ms)

        DispatchQueue.main.async { [weak self] in
            self?.updateBoundingBoxes(with: dets)
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            self?.statsLabel.string = statsText
            CATransaction.commit()
        }

        // Track the best detection since last display update.
        if let top = dets.max(by: { $0.confidence < $1.confidence }) {
            if top.confidence > bestConf {
                bestLabel = top.label
                bestConf = top.confidence
            }
        }

        // Update the displayed central label only every displayInterval seconds.
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastDisplayUpdate >= displayInterval else { return }
        lastDisplayUpdate = now

        let displayDenom = Self.extractDenomination(from: bestLabel)
        let displayTE = Self.extractTahunEmisi(from: bestLabel)

        // Reset accumulator for next interval.
        bestLabel = ""
        bestConf = 0

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }

            if !displayDenom.isEmpty {
                self.denomLabel.text = displayDenom
                self.tahunEmisiLabel.text = displayTE.isEmpty ? "" : "Tahun Emisi: \(displayTE)"
                self.tahunEmisiLabel.isHidden = displayTE.isEmpty

                if self.denomLabel.alpha < 1 {
                    UIView.animate(withDuration: 0.2) {
                        self.denomLabel.alpha = 1
                        self.tahunEmisiLabel.alpha = 1
                    }
                }
                // Quick scale pop animation on each update.
                self.denomLabel.transform = CGAffineTransform(scaleX: 0.85, y: 0.85)
                self.tahunEmisiLabel.transform = CGAffineTransform(scaleX: 0.85, y: 0.85)
                UIView.animate(withDuration: 0.25, delay: 0,
                               usingSpringWithDamping: 0.6,
                               initialSpringVelocity: 0.5) {
                    self.denomLabel.transform = .identity
                    self.tahunEmisiLabel.transform = .identity
                }
            } else {
                UIView.animate(withDuration: 0.3) {
                    self.denomLabel.alpha = 0
                    self.tahunEmisiLabel.alpha = 0
                }
            }
        }
    }

    // MARK: - Update Real-Time Bounding Boxes

    private func updateBoundingBoxes(with dets: [Detection]) {
        let viewW = view.bounds.width
        let viewH = view.bounds.height
        guard viewW > 0, viewH > 0 else { return }

        let squareSide = min(viewW, viewH)
        let offsetX = (viewW - squareSide) / 2
        let offsetY = (viewH - squareSide) / 2

        for i in 0..<boxViews.count {
            guard i < dets.count else {
                boxViews[i].hide()
                continue
            }
            let det = dets[i]
            let screenRect = CGRect(
                x: offsetX + det.normRect.minX * squareSide,
                y: offsetY + det.normRect.minY * squareSide,
                width: det.normRect.width * squareSide,
                height: det.normRect.height * squareSide
            )

            let denom = Self.extractDenomination(from: det.label)
            let te = Self.extractTahunEmisi(from: det.label)
            let teSuffix = te.isEmpty ? "" : " [\(te)]"
            let tagText = "\(denom)\(teSuffix) \(Int(det.confidence * 100))%"

            let color = detector.colors[det.classIndex % detector.colors.count]
            boxViews[i].show(frame: screenRect, label: tagText, color: color, alpha: CGFloat(det.confidence))
        }
    }

    // MARK: - Extract denomination from label

    /// "kertas-1000-te2000" → "1.000"
    /// "koin-500-te2016" → "500"
    static func extractDenomination(from label: String) -> String {
        guard !label.isEmpty else { return "" }
        if label == "palm_crown" { return "Palm Crown" }
        let parts = label.split(separator: "-")
        guard parts.count >= 2, let value = Int(parts[1]) else {
            return label.replacingOccurrences(of: "_", with: " ").capitalized
        }
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.groupingSeparator = "."
        return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }

    // MARK: - Extract Tahun Emisi from label

    /// "kertas-1000-te2000" → "TE 2000"
    /// "kertas-100000-te2022" → "TE 2022"
    /// "koin-500-te2016" → "TE 2016"
    static func extractTahunEmisi(from label: String) -> String {
        guard !label.isEmpty else { return "" }
        let parts = label.split(separator: "-")
        guard parts.count >= 3 else { return "" }
        let rawTE = parts[2].uppercased() // e.g. "TE2000"
        if rawTE.hasPrefix("TE") {
            let year = rawTE.dropFirst(2)
            return "TE \(year)"
        }
        return String(rawTE)
    }
}
#elseif canImport(AppKit)
final class FlippedView: NSView {
    override var isFlipped: Bool { true }
}

final class CameraVC: NSViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let detector: Detector
    private let modeTitle: String
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "session")
    private let inferenceQueue = DispatchQueue(label: "inference")
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var isProcessing = false

    // Real-time bounding box pool
    private var boxViews: [BoundingBoxView] = []

    // Header badge
    private let modeBadge = NSTextField(labelWithString: "")

    // Big denomination label (updated every ~1 second)
    private let denomLabel = NSTextField(labelWithString: "")
    // Tahun Emisi label below the denomination
    private let tahunEmisiLabel = NSTextField(labelWithString: "")

    private var lastDisplayUpdate: CFAbsoluteTime = 0
    private let displayInterval: CFAbsoluteTime = 1.0

    // Accumulate the best detection across frames between display updates
    private var bestLabel: String = ""
    private var bestConf: Float = 0

    // Stats
    private let statsLabel = CATextLayer()

    init(detector: Detector, modeTitle: String = "Cash") {
        self.detector = detector
        self.modeTitle = modeTitle
        super.init(nibName: nil, bundle: nil)
    }
    required init?(coder: NSCoder) { fatalError() }

    override func loadView() {
        let v = FlippedView()
        v.wantsLayer = true
        self.view = v
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        guard let hostLayer = view.layer else { return }

        // Camera preview
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        hostLayer.addSublayer(previewLayer)

        // Add bounding box views to preview layer (drawn behind foreground labels)
        for _ in 0..<30 {
            let bv = BoundingBoxView()
            bv.addToLayer(previewLayer)
            boxViews.append(bv)
        }

        // Header mode badge
        modeBadge.stringValue = "Identifying \(modeTitle) »"
        modeBadge.alignment = .center
        modeBadge.font = NSFont.systemFont(ofSize: 14, weight: .semibold)
        modeBadge.textColor = .white
        modeBadge.backgroundColor = NSColor.black.withAlphaComponent(0.6)
        modeBadge.wantsLayer = true
        modeBadge.layer?.cornerRadius = 14
        modeBadge.layer?.masksToBounds = true
        view.addSubview(modeBadge)

        // Big denomination label
        denomLabel.alignment = .center
        denomLabel.maximumNumberOfLines = 1
        denomLabel.font = NSFont.systemFont(ofSize: 96, weight: .black)
        denomLabel.textColor = .systemYellow
        denomLabel.wantsLayer = true
        denomLabel.alphaValue = 0
        view.addSubview(denomLabel)

        // Tahun Emisi label
        tahunEmisiLabel.alignment = .center
        tahunEmisiLabel.font = NSFont.systemFont(ofSize: 18, weight: .bold)
        tahunEmisiLabel.textColor = .white
        tahunEmisiLabel.backgroundColor = NSColor.black.withAlphaComponent(0.6)
        tahunEmisiLabel.wantsLayer = true
        tahunEmisiLabel.layer?.cornerRadius = 14
        tahunEmisiLabel.layer?.masksToBounds = true
        tahunEmisiLabel.alphaValue = 0
        view.addSubview(tahunEmisiLabel)

        // Stats overlay
        let scale = NSScreen.main?.backingScaleFactor ?? 2.0
        statsLabel.fontSize = 12
        statsLabel.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .medium)
        statsLabel.foregroundColor = NSColor.white.cgColor
        statsLabel.backgroundColor = NSColor.black.withAlphaComponent(0.5).cgColor
        statsLabel.cornerRadius = 8
        statsLabel.masksToBounds = true
        statsLabel.contentsScale = scale
        statsLabel.alignmentMode = .center
        hostLayer.addSublayer(statsLabel)

        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard ok else { return }
            self?.sessionQueue.async { self?.setupCamera() }
        }
    }

    override func viewDidLayout() {
        super.viewDidLayout()
        let bounds = view.bounds
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        previewLayer.frame = bounds
        CATransaction.commit()

        modeBadge.frame = CGRect(
            x: (bounds.width - 250) / 2,
            y: 16,
            width: 250,
            height: 28
        )

        statsLabel.frame = CGRect(
            x: (bounds.width - 180) / 2,
            y: 50,
            width: 180,
            height: 22
        )

        let pad: CGFloat = 20
        let denomY = bounds.height * 0.28
        let denomH = max(bounds.height * 0.24, 100)
        denomLabel.frame = CGRect(
            x: pad,
            y: denomY,
            width: bounds.width - pad * 2,
            height: denomH
        )

        tahunEmisiLabel.frame = CGRect(
            x: (bounds.width - 240) / 2,
            y: denomY + denomH + 8,
            width: 240,
            height: 32
        )
    }

    override func viewWillAppear() {
        super.viewWillAppear()
        sessionQueue.async { if !self.session.isRunning { self.session.startRunning() } }
    }

    override func viewWillDisappear() {
        super.viewWillDisappear()
        sessionQueue.async { self.session.stopRunning() }
    }

    private func setupCamera() {
        session.beginConfiguration()
        session.sessionPreset = .high
        let dev = AVCaptureDevice.default(for: .video)
            ?? AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external], mediaType: .video, position: .unspecified).devices.first
        guard let dev, let input = try? AVCaptureDeviceInput(device: dev) else {
            session.commitConfiguration()
            return
        }
        if session.canAddInput(input) { session.addInput(input) }
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: inferenceQueue)
        if session.canAddOutput(out) { session.addOutput(out) }
        session.commitConfiguration()
        session.startRunning()
    }

    // MARK: - Inference (every frame)

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from conn: AVCaptureConnection) {
        guard !isProcessing else { return }
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }

        isProcessing = true
        let start = CACurrentMediaTime()
        let dets = detector.detect(pixelBuffer: pb)
        let ms = (CACurrentMediaTime() - start) * 1000
        isProcessing = false

        // Update real-time bounding boxes and stats on background
        let statsText = String(format: "  %.0f ms  ", ms)

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.updateBoundingBoxes(with: dets)
            CATransaction.begin()
            CATransaction.setDisableActions(true)
            self.statsLabel.string = statsText
            CATransaction.commit()
        }

        // Track the best detection since last display update.
        if let top = dets.max(by: { $0.confidence < $1.confidence }) {
            if top.confidence > bestConf {
                bestLabel = top.label
                bestConf = top.confidence
            }
        }

        // Update the displayed central label only every displayInterval seconds.
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastDisplayUpdate >= displayInterval else { return }
        lastDisplayUpdate = now

        let displayDenom = Self.extractDenomination(from: bestLabel)
        let displayTE = Self.extractTahunEmisi(from: bestLabel)

        // Reset accumulator for next interval.
        bestLabel = ""
        bestConf = 0

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }

            if !displayDenom.isEmpty {
                self.denomLabel.stringValue = displayDenom
                self.tahunEmisiLabel.stringValue = displayTE.isEmpty ? "" : "Tahun Emisi: \(displayTE)"
                self.tahunEmisiLabel.isHidden = displayTE.isEmpty

                if self.denomLabel.alphaValue < 1 {
                    NSAnimationContext.runAnimationGroup { ctx in
                        ctx.duration = 0.2
                        self.denomLabel.animator().alphaValue = 1
                        self.tahunEmisiLabel.animator().alphaValue = 1
                    }
                }
            } else {
                NSAnimationContext.runAnimationGroup { ctx in
                    ctx.duration = 0.3
                    self.denomLabel.animator().alphaValue = 0
                    self.tahunEmisiLabel.animator().alphaValue = 0
                }
            }
        }
    }

    // MARK: - Update Real-Time Bounding Boxes

    private func updateBoundingBoxes(with dets: [Detection]) {
        let viewW = view.bounds.width
        let viewH = view.bounds.height
        guard viewW > 0, viewH > 0 else { return }

        let squareSide = min(viewW, viewH)
        let offsetX = (viewW - squareSide) / 2
        let offsetY = (viewH - squareSide) / 2

        for i in 0..<boxViews.count {
            guard i < dets.count else {
                boxViews[i].hide()
                continue
            }
            let det = dets[i]
            let screenRect = CGRect(
                x: offsetX + det.normRect.minX * squareSide,
                y: offsetY + det.normRect.minY * squareSide,
                width: det.normRect.width * squareSide,
                height: det.normRect.height * squareSide
            )

            let denom = Self.extractDenomination(from: det.label)
            let te = Self.extractTahunEmisi(from: det.label)
            let teSuffix = te.isEmpty ? "" : " [\(te)]"
            let tagText = "\(denom)\(teSuffix) \(Int(det.confidence * 100))%"

            let color = detector.colors[det.classIndex % detector.colors.count]
            boxViews[i].show(frame: screenRect, label: tagText, color: color, alpha: CGFloat(det.confidence))
        }
    }

    // MARK: - Extract denomination from label

    /// "kertas-1000-te2000" → "1.000"
    /// "koin-500-te2016" → "500"
    static func extractDenomination(from label: String) -> String {
        guard !label.isEmpty else { return "" }
        if label == "palm_crown" { return "Palm Crown" }
        let parts = label.split(separator: "-")
        guard parts.count >= 2, let value = Int(parts[1]) else {
            return label.replacingOccurrences(of: "_", with: " ").capitalized
        }
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.groupingSeparator = "."
        return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }

    // MARK: - Extract Tahun Emisi from label

    /// "kertas-1000-te2000" → "TE 2000"
    /// "kertas-100000-te2022" → "TE 2022"
    /// "koin-500-te2016" → "TE 2016"
    static func extractTahunEmisi(from label: String) -> String {
        guard !label.isEmpty else { return "" }
        let parts = label.split(separator: "-")
        guard parts.count >= 3 else { return "" }
        let rawTE = parts[2].uppercased() // e.g. "TE2000"
        if rawTE.hasPrefix("TE") {
            let year = rawTE.dropFirst(2)
            return "TE \(year)"
        }
        return String(rawTE)
    }
}

struct CameraVCWrapper: NSViewControllerRepresentable {
    let detector: Detector
    var modeTitle: String = "Palm Crown"

    func makeNSViewController(context: Context) -> CameraVC {
        CameraVC(detector: detector, modeTitle: modeTitle)
    }
    func updateNSViewController(_ vc: CameraVC, context: Context) {}
}
#endif

// MARK: - Bounding Box View (CALayer pool for camera)

class BoundingBoxView {
    let shapeLayer = CAShapeLayer()
    let fillLayer = CAShapeLayer()
    let textLayer = CATextLayer()

    init() {
        shapeLayer.fillColor = nil
        shapeLayer.lineWidth = 2.5
        shapeLayer.lineCap = .round
        shapeLayer.lineJoin = .round
        shapeLayer.isHidden = true

        fillLayer.isHidden = true

        textLayer.fontSize = 11
        #if canImport(UIKit)
        textLayer.font = UIFont.systemFont(ofSize: 11, weight: .bold)
        textLayer.contentsScale = UIScreen.main.scale
        #elseif canImport(AppKit)
        textLayer.font = NSFont.boldSystemFont(ofSize: 11)
        textLayer.contentsScale = NSScreen.main?.backingScaleFactor ?? 2.0
        #endif
        textLayer.foregroundColor = PlatformColor.white.cgColor
        textLayer.isHidden = true
        textLayer.cornerRadius = 6
        textLayer.masksToBounds = true
        textLayer.alignmentMode = .center
    }

    func addToLayer(_ parent: CALayer) {
        parent.addSublayer(fillLayer)
        parent.addSublayer(shapeLayer)
        parent.addSublayer(textLayer)
    }

    func show(frame: CGRect, label: String, color: PlatformColor, alpha: CGFloat) {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        let path = CGPath(roundedRect: frame, cornerWidth: 8, cornerHeight: 8, transform: nil)
        shapeLayer.path = path
        shapeLayer.strokeColor = color.cgColor
        shapeLayer.isHidden = false

        fillLayer.path = path
        fillLayer.fillColor = color.withAlphaComponent(0.12).cgColor
        fillLayer.isHidden = false

        textLayer.string = "  \(label)  "
        textLayer.backgroundColor = color.cgColor
        let tw = CGFloat(label.count) * 6.5 + 16
        let ty = frame.minY > 24 ? frame.minY - 20 : frame.maxY + 4
        textLayer.frame = CGRect(x: frame.minX, y: ty,
                                 width: min(tw, max(frame.width + 60, 180)), height: 18)
        textLayer.isHidden = false
        CATransaction.commit()
    }

    func hide() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        shapeLayer.isHidden = true
        fillLayer.isHidden = true
        textLayer.isHidden = true
        CATransaction.commit()
    }
}

#Preview { ContentView() }
