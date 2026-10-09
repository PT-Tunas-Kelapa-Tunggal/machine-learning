#if canImport(UIKit)
import UIKit
public typealias PlatformImage = UIImage
public typealias PlatformColor = UIColor
#elseif canImport(AppKit)
import AppKit
public typealias PlatformImage = NSImage
public typealias PlatformColor = NSColor

extension NSImage {
    var cgImage: CGImage? {
        var proposedRect = CGRect(origin: .zero, size: size)
        return cgImage(forProposedRect: &proposedRect, context: nil, hints: nil)
    }
}
#endif
import CoreML
import Vision

// MARK: - Detection Result

struct Detection: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
    let classIndex: Int
    let normRect: CGRect // [0,1], top-left origin (x, y, w, h)
    var trackId: Int? = nil
    // Recent Kalman-center history (normalized top-left, oldest → newest).
    // Populated by the tracker for visualization trails; empty for raw detections.
    var trail: [CGPoint] = []
}

// MARK: - Detector

class Detector: ObservableObject {
    private var mlModel: MLModel?
    private var vnModel: VNCoreMLModel?
    @Published var isReady = false

    let modelName: String
    let labels: [String]
    @Published var confThreshold: Float = 0.35
    @Published var maxDetections: Int = 30

    let colors: [PlatformColor] = [
        PlatformColor(red: 0.0, green: 0.48, blue: 1.0, alpha: 1.0), // Solid Blue
    ]

    // MARK: - Labels
    
    static let kropsLabels = [
        "palm_crown"
    ]

    static let cashLabels = [
        "kertas-1000-te2000",
        "kertas-1000-te2016",
        "kertas-1000-te2022",
        "kertas-10000-te2005",
        "kertas-10000-te2016",
        "kertas-10000-te2022",
        "kertas-100000-te2014",
        "kertas-100000-te2016",
        "kertas-100000-te2022",
        "kertas-2000-te2009",
        "kertas-2000-te2016",
        "kertas-2000-te2022",
        "kertas-20000-te2004",
        "kertas-20000-te2016",
        "kertas-20000-te2022",
        "kertas-5000-te2001",
        "kertas-5000-te2016",
        "kertas-5000-te2022",
        "kertas-50000-te2005",
        "kertas-50000-te2016",
        "kertas-50000-te2022"
    ]

    static let coinLabels = [
        "koin-100-te1999",
        "koin-100-te2016",
        "koin-1000-te2010",
        "koin-1000-te2016",
        "koin-200-te2003",
        "koin-200-te2016",
        "koin-500-te2002",
        "koin-500-te2003",
        "koin-500-te2016"
    ]

    static let allLabels = kropsLabels
    static let defaultLabels = kropsLabels

    init(modelName: String = "yolo26m-krops-v0.1", labels: [String] = Detector.kropsLabels) {
        self.modelName = modelName
        self.labels = labels
        loadModel(named: modelName)
    }

    /// Load the specified CoreML model.
    private func loadModel(named targetName: String) {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all

        // 1. Try direct bundle resource URL
        if let url = Bundle.main.url(forResource: targetName, withExtension: "mlmodelc") {
            if loadModel(at: url, configuration: cfg) { return }
        }

        // 2. Scan resource directory for model matching targetName
        guard let resourcePath = Bundle.main.resourcePath else { return }
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(atPath: resourcePath) else { return }

        for item in items where item.hasSuffix(".mlmodelc") && item.contains(targetName) {
            let url = URL(fileURLWithPath: resourcePath).appendingPathComponent(item)
            if loadModel(at: url, configuration: cfg) { return }
        }

        // 3. Fallback: match by model prefix or any .mlmodelc
        for item in items where item.hasSuffix(".mlmodelc") {
            if item.contains("krops") || item.contains("yolo") {
                let url = URL(fileURLWithPath: resourcePath).appendingPathComponent(item)
                if loadModel(at: url, configuration: cfg) { return }
            }
        }
        for item in items where item.hasSuffix(".mlmodelc") {
            let url = URL(fileURLWithPath: resourcePath).appendingPathComponent(item)
            if loadModel(at: url, configuration: cfg) { return }
        }
    }

    private func loadModel(at url: URL, configuration: MLModelConfiguration) -> Bool {
        do {
            let model = try MLModel(contentsOf: url, configuration: configuration)
            mlModel = model
            vnModel = try VNCoreMLModel(for: model)
            DispatchQueue.main.async { self.isReady = true }
            print("[Detector] Loaded model: \(url.lastPathComponent)")
            return true
        } catch {
            print("[Detector] Model load error (\(url.lastPathComponent)): \(error)")
            return false
        }
    }

    var visionModel: VNCoreMLModel? { vnModel }

    // MARK: - Detect on PlatformImage (async, for photo mode)

    func detect(image: PlatformImage) async -> [Detection] {
        guard let cgImage = image.cgImage else { return [] }
        return await withCheckedContinuation { cont in
            guard let vnModel else { cont.resume(returning: []); return }
            let req = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
                cont.resume(returning: self?.parseResults(req) ?? [])
            }
            req.imageCropAndScaleOption = .centerCrop
            try? VNImageRequestHandler(cgImage: cgImage, orientation: .up).perform([req])
        }
    }

    // MARK: - Detect on CVPixelBuffer (sync, for video/camera)

    func detect(pixelBuffer: CVPixelBuffer) -> [Detection] {
        guard let vnModel else { return [] }
        var result: [Detection] = []
        let req = VNCoreMLRequest(model: vnModel) { [weak self] req, _ in
            result = self?.parseResults(req) ?? []
        }
        req.imageCropAndScaleOption = .centerCrop
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up).perform([req])
        return result
    }

    // MARK: - Parse

    func parseResults(_ req: VNRequest) -> [Detection] {
        if let results = req.results as? [VNRecognizedObjectObservation], !results.isEmpty {
            let detections: [Detection] = results.compactMap { obs in
                guard let top = obs.labels.first, top.confidence >= confThreshold else { return nil }
                let vr = obs.boundingBox
                let idx = self.labels.firstIndex(of: top.identifier) ?? 0
                return Detection(label: top.identifier, confidence: top.confidence, classIndex: idx,
                                 normRect: CGRect(x: vr.minX, y: 1 - vr.maxY, width: vr.width, height: vr.height))
            }
            let sorted = detections.sorted(by: { $0.confidence > $1.confidence })
            return maxDetections > 0 ? Array(sorted.prefix(maxDetections)) : sorted
        }

        guard let results = req.results as? [VNCoreMLFeatureValueObservation] else { return [] }
        for obs in results {
            guard let arr = obs.featureValue.multiArrayValue else { continue }
            let shape = arr.shape.map { $0.intValue }

            // Format 1: [1, N, 6] (End-to-end NMS output: x1, y1, x2, y2, conf, class_id)
            if shape.count == 3 && shape[2] == 6 {
                var out: [Detection] = []
                for i in 0..<shape[1] {
                    let conf = arr[[0, i, 4] as [NSNumber]].floatValue
                    guard conf >= confThreshold else { continue }
                    let x1 = CGFloat(arr[[0, i, 0] as [NSNumber]].floatValue) / 640
                    let y1 = CGFloat(arr[[0, i, 1] as [NSNumber]].floatValue) / 640
                    let x2 = CGFloat(arr[[0, i, 2] as [NSNumber]].floatValue) / 640
                    let y2 = CGFloat(arr[[0, i, 3] as [NSNumber]].floatValue) / 640
                    let cid = Int(arr[[0, i, 5] as [NSNumber]].floatValue)
                    let label = cid < self.labels.count ? self.labels[cid] : "\(cid)"
                    out.append(Detection(label: label, confidence: conf, classIndex: cid,
                                         normRect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)))
                }
                let sorted = out.sorted(by: { $0.confidence > $1.confidence })
                return maxDetections > 0 ? Array(sorted.prefix(maxDetections)) : sorted
            }

            // Format 2: [1, 4 + numClasses, numAnchors] (Raw YOLO output e.g. [1, 5, 8400])
            if shape.count == 3 && shape[1] < shape[2] {
                let channels = shape[1]
                let numAnchors = shape[2]
                let numClasses = channels - 4
                guard numClasses > 0 else { continue }

                var candidates: [(rect: CGRect, conf: Float, classIdx: Int)] = []
                let strides = arr.strides.map { $0.intValue }
                let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: arr.count)
                let cStride = strides[1]
                let aStride = strides[2]

                for i in 0..<numAnchors {
                    let offset = i * aStride
                    var maxConf: Float = 0
                    var maxClass = 0
                    for c in 0..<numClasses {
                        let conf = ptr[offset + (4 + c) * cStride]
                        if conf > maxConf {
                            maxConf = conf
                            maxClass = c
                        }
                    }
                    guard maxConf >= confThreshold else { continue }

                    let cx = CGFloat(ptr[offset + 0 * cStride])
                    let cy = CGFloat(ptr[offset + 1 * cStride])
                    let w  = CGFloat(ptr[offset + 2 * cStride])
                    let h  = CGFloat(ptr[offset + 3 * cStride])

                    let x1 = (cx - w / 2) / 640
                    let y1 = (cy - h / 2) / 640
                    let nw = w / 640
                    let nh = h / 640

                    let rect = CGRect(
                        x: max(0, min(1, x1)),
                        y: max(0, min(1, y1)),
                        width: min(1, nw),
                        height: min(1, nh)
                    )
                    candidates.append((rect: rect, conf: maxConf, classIdx: maxClass))
                }

                // Fast Non-Maximum Suppression (NMS)
                candidates.sort { $0.conf > $1.conf }
                var selected: [Detection] = []
                let iouThreshold: CGFloat = 0.45

                for cand in candidates {
                    var keep = true
                    for s in selected {
                        let inter = cand.rect.intersection(s.normRect)
                        let interArea = max(0, inter.width) * max(0, inter.height)
                        let unionArea = cand.rect.width * cand.rect.height + s.normRect.width * s.normRect.height - interArea
                        let iou = unionArea > 0 ? interArea / unionArea : 0
                        if iou > iouThreshold {
                            keep = false
                            break
                        }
                    }
                    if keep {
                        let label = cand.classIdx < self.labels.count ? self.labels[cand.classIdx] : "\(cand.classIdx)"
                        selected.append(Detection(label: label, confidence: cand.conf, classIndex: cand.classIdx, normRect: cand.rect))
                        if maxDetections > 0 && selected.count >= maxDetections {
                            break
                        }
                    }
                }
                return selected
            }
        }
        return []
    }

    // MARK: - PlatformImage → CVPixelBuffer

    static func imageToPixelBuffer(_ image: PlatformImage, size: CGSize = CGSize(width: 640, height: 640)) -> CVPixelBuffer? {
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, Int(size.width), Int(size.height),
                            kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard let pixelBuffer = pb else { return nil }
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pixelBuffer),
                            width: Int(size.width), height: Int(size.height),
                            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let context = ctx, let cgImage = image.cgImage else {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            return nil
        }
        #if canImport(UIKit)
        // Normalize EXIF orientation
        let renderer = UIGraphicsImageRenderer(size: size)
        let normalized = renderer.image { _ in image.draw(in: CGRect(origin: .zero, size: size)) }
        if let normalizedCG = normalized.cgImage {
            context.draw(normalizedCG, in: CGRect(origin: .zero, size: size))
        } else {
            context.draw(cgImage, in: CGRect(origin: .zero, size: size))
        }
        #elseif canImport(AppKit)
        context.draw(cgImage, in: CGRect(origin: .zero, size: size))
        #endif
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
        return pixelBuffer
    }
}
