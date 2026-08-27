import AppKit
import Foundation
import Vision

struct OCRLine: Codable {
    let text: String
    let confidence: Float
    let x: Double
    let y: Double
}

struct OCRPayload: Codable {
    let lines: [OCRLine]
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

guard CommandLine.arguments.count == 2 else {
    fail("usage: macos_vision_ocr <image-path>")
}

let imagePath = CommandLine.arguments[1]
guard
    let image = NSImage(contentsOfFile: imagePath),
    let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil)
else {
    fail("unable to read image")
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["en-US"]

do {
    try VNImageRequestHandler(cgImage: cgImage).perform([request])
} catch {
    fail("vision request failed: \(error.localizedDescription)")
}

let observations = request.results ?? []
let lines = observations.compactMap { observation -> OCRLine? in
    guard let candidate = observation.topCandidates(1).first else {
        return nil
    }
    return OCRLine(
        text: candidate.string,
        confidence: candidate.confidence,
        x: observation.boundingBox.minX,
        y: observation.boundingBox.maxY
    )
}.sorted {
    if abs($0.y - $1.y) > 0.02 {
        return $0.y > $1.y
    }
    return $0.x < $1.x
}

do {
    let data = try JSONEncoder().encode(OCRPayload(lines: lines))
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    fail("unable to encode OCR output")
}
