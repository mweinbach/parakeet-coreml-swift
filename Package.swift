// swift-tools-version:5.10
import PackageDescription

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
            path: "Sources/ParakeetTDT"
        ),
        .executableTarget(
            name: "ParakeetCLI",
            dependencies: [
                "ParakeetTDT",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/ParakeetCLI"
        ),
        .testTarget(
            name: "ParakeetTDTTests",
            dependencies: ["ParakeetTDT"],
            path: "Tests/ParakeetTDTTests"
        ),
    ]
)
