// swift-tools-version:5.10
import PackageDescription

// ParakeetTDT does real-time ML inference -- running it at `-Onone` turns a
// 400x RTFx pipeline into a 40x one. Rather than making every consumer
// discover this the hard way, the library declares `-O` for its own source
// unconditionally. App code keeps its own Debug/Release optimization
// settings; only the library's own code is force-optimized.
//
// The `.unsafeFlags` marker is the only way to express "this target needs
// a specific optimization level regardless of the build configuration" from
// Package.swift in SwiftPM 5.10+.
let parakeetForceOptimization: [SwiftSetting] = [
    .unsafeFlags(["-O"], .when(configuration: .debug)),
]

let package = Package(
    name: "ParakeetCoreML",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(
            name: "ParakeetTDT",
            targets: ["ParakeetTDT"]
        ),
        .executable(
            name: "parakeet",
            targets: ["ParakeetCLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "ParakeetTDT",
            path: "Sources/ParakeetTDT",
            swiftSettings: parakeetForceOptimization
        ),
        .executableTarget(
            name: "ParakeetCLI",
            dependencies: [
                "ParakeetTDT",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/ParakeetCLI",
            swiftSettings: parakeetForceOptimization
        ),
        .testTarget(
            name: "ParakeetTDTTests",
            dependencies: ["ParakeetTDT"],
            path: "Tests/ParakeetTDTTests"
        ),
    ]
)
