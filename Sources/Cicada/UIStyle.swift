import AppKit
import SwiftUI

/// Dynamic UI typography and layout scaling. On high-resolution non-Retina displays (like 2560×1440 1x)
/// default 11–13pt text looks microscopic; scaling keeps notes and panels comfortable to read.
enum UIStyle {
    static var scale: CGFloat = 1.25

    static func update(_ mode: String) {
        switch mode {
        case "normal": scale = 1.0
        case "xlarge": scale = 1.40
        default: scale = 1.25 // "large"
        }
    }

    static func pt(_ val: CGFloat) -> CGFloat {
        round(val * scale)
    }

    static func font(_ size: CGFloat, weight: Font.Weight = .regular, design: Font.Design = .default) -> Font {
        .system(size: round(size * scale), weight: weight, design: design)
    }
}

extension Font {
    static func ui(_ size: CGFloat, weight: Font.Weight = .regular, design: Font.Design = .default) -> Font {
        UIStyle.font(size, weight: weight, design: design)
    }
}
