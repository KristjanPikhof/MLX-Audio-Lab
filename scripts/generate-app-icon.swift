#!/usr/bin/env swift
import AppKit
import Foundation

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let resources = root.appending(path: "Sources/MLXAudioLab/Resources", directoryHint: .isDirectory)
let generated = root.appending(path: ".generated", directoryHint: .isDirectory)
let iconset = generated.appending(path: "AppIcon.iconset", directoryHint: .isDirectory)
let output = resources.appending(path: "AppIcon.icns")

try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
try? FileManager.default.removeItem(at: generated)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)

let iconFiles: [(name: String, pixels: Int)] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024)
]

func drawIcon(pixels: Int) throws -> Data {
    let size = NSSize(width: pixels, height: pixels)
    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: pixels,
        pixelsHigh: pixels,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        throw NSError(domain: "IconGenerator", code: 1)
    }

    bitmap.size = size
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)

    let scale = CGFloat(pixels) / 1024
    func rect(_ x: CGFloat, _ y: CGFloat, _ width: CGFloat, _ height: CGFloat) -> NSRect {
        NSRect(x: x * scale, y: y * scale, width: width * scale, height: height * scale)
    }
    func point(_ x: CGFloat, _ y: CGFloat) -> NSPoint {
        NSPoint(x: x * scale, y: y * scale)
    }

    NSColor.clear.setFill()
    NSRect(origin: .zero, size: size).fill()

    let bgPath = NSBezierPath(
        roundedRect: rect(64, 64, 896, 896),
        xRadius: 208 * scale,
        yRadius: 208 * scale
    )
    NSGradient(colors: [
        NSColor(red: 0.067, green: 0.094, blue: 0.153, alpha: 1),
        NSColor(red: 0.059, green: 0.463, blue: 0.431, alpha: 1),
        NSColor(red: 0.976, green: 0.451, blue: 0.086, alpha: 1)
    ])?.draw(in: bgPath, angle: -45)

    NSColor(red: 0.043, green: 0.071, blue: 0.125, alpha: 0.42).setFill()
    NSBezierPath(ovalIn: rect(302, 216, 420, 420)).fill()

    let wave = NSBezierPath()
    wave.move(to: point(254, 560))
    wave.curve(to: point(408, 560), controlPoint1: point(306, 424), controlPoint2: point(356, 424))
    wave.curve(to: point(562, 560), controlPoint1: point(460, 696), controlPoint2: point(510, 696))
    wave.curve(to: point(770, 560), controlPoint1: point(614, 424), controlPoint2: point(666, 424))
    wave.lineWidth = 54 * scale
    wave.lineCapStyle = .round
    NSColor(red: 0.70, green: 0.95, blue: 0.94, alpha: 1).setStroke()
    wave.stroke()

    NSColor(red: 0.90, green: 0.98, blue: 1.0, alpha: 0.94).setFill()
    NSBezierPath(roundedRect: rect(256, 440, 168, 280), xRadius: 58 * scale, yRadius: 58 * scale).fill()
    NSBezierPath(roundedRect: rect(600, 440, 168, 280), xRadius: 58 * scale, yRadius: 58 * scale).fill()

    let mlxFont = NSFont(name: "AvenirNext-Heavy", size: 148 * scale)
        ?? NSFont.boldSystemFont(ofSize: 148 * scale)
    let audioFont = NSFont(name: "AvenirNext-Bold", size: 72 * scale)
        ?? NSFont.boldSystemFont(ofSize: 72 * scale)
    let paragraph = NSMutableParagraphStyle()
    paragraph.alignment = .center

    ("MLX" as NSString).draw(in: rect(0, 640, 1024, 180), withAttributes: [
        .font: mlxFont,
        .foregroundColor: NSColor.white,
        .paragraphStyle: paragraph,
        .kern: 8 * scale
    ])

    ("AUDIO" as NSString).draw(in: rect(0, 174, 1024, 96), withAttributes: [
        .font: audioFont,
        .foregroundColor: NSColor(white: 1, alpha: 0.92),
        .paragraphStyle: paragraph
    ])

    NSGraphicsContext.restoreGraphicsState()

    guard let png = bitmap.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "IconGenerator", code: 2)
    }
    return png
}

for file in iconFiles {
    let data = try drawIcon(pixels: file.pixels)
    try data.write(to: iconset.appending(path: file.name))
}

let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconset.path, "-o", output.path]
try process.run()
process.waitUntilExit()

guard process.terminationStatus == 0 else {
    throw NSError(domain: "IconGenerator", code: Int(process.terminationStatus))
}

print("Generated \(output.path)")
