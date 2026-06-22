// swift-tools-version:6.2
import Foundation
import PackageDescription

let packageRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
let infoPlistPath = "\(packageRoot)/Sources/MLXAudioLab/Info.plist"

let package = Package(
    name: "MLXAudioLab",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "MLXAudioLab", targets: ["MLXAudioLab"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/Blaizzy/mlx-audio-swift.git",
            revision: "3f6b0553188a921f635df54b5e20442001037336"
        ),
        .package(
            url: "https://github.com/huggingface/swift-huggingface.git",
            .upToNextMajor(from: "0.8.1")
        )
    ],
    targets: [
        .executableTarget(
            name: "MLXAudioLab",
            dependencies: [
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift")
            ],
            exclude: ["Info.plist"],
            resources: [
                .copy("Resources/AppIcon.icns")
            ],
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", infoPlistPath
                ])
            ]
        )
    ]
)
