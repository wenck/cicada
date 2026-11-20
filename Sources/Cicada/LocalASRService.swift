import Foundation

/// Manages the local SenseVoice/Whisper ASR daemon process alongside Cicada.
/// Automatically starts it on launch if the local model exists, and terminates it cleanly when Cicada exits.
final class LocalASRService: @unchecked Sendable {
    static let shared = LocalASRService()
    private var process: Process?

    private let modelDir = NSString(string: "~/.cache/asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17").expandingTildeInPath
    private let scriptPath = NSString(string: "~/.cache/asr_models/local_asr_service.py").expandingTildeInPath

    var isModelInstalled: Bool {
        let modelFile = (modelDir as NSString).appendingPathComponent("model.int8.onnx")
        return FileManager.default.fileExists(atPath: modelFile) && FileManager.default.fileExists(atPath: scriptPath)
    }

    func startIfNeeded() {
        guard isModelInstalled else {
            debugLog("local-asr: model or script not found, skip auto-start")
            return
        }

        Task {
            if await isServiceResponding() {
                debugLog("local-asr: service is already running on port 8000")
                return
            }
            launchProcess()
        }
    }

    private func isServiceResponding() async -> Bool {
        guard let url = URL(string: "http://127.0.0.1:8000/health") else { return false }
        var req = URLRequest(url: url)
        req.timeoutInterval = 0.5
        do {
            let (_, resp) = try await URLSession.shared.data(for: req)
            return (resp as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    private func launchProcess() {
        let p = Process()
        let uvCandidates = ["/opt/homebrew/bin/uv", "/usr/local/bin/uv"]
        let uvPath = uvCandidates.first(where: { FileManager.default.fileExists(atPath: $0) })

        if let uvPath {
            p.executableURL = URL(fileURLWithPath: uvPath)
            p.arguments = ["run", scriptPath]
        } else {
            p.executableURL = URL(fileURLWithPath: "/bin/zsh")
            p.arguments = ["-l", "-c", "uv run \(scriptPath)"]
        }

        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + (env["PATH"] ?? "")
        p.environment = env

        let logPath = "/tmp/cicada-local-asr.log"
        FileManager.default.createFile(atPath: logPath, contents: nil)
        if let fileHandle = FileHandle(forWritingAtPath: logPath) {
            p.standardOutput = fileHandle
            p.standardError = fileHandle
        } else {
            p.standardOutput = FileHandle.nullDevice
            p.standardError = FileHandle.nullDevice
        }

        do {
            try p.run()
            self.process = p
            debugLog("local-asr: launched daemon (pid: \(p.processIdentifier))")
        } catch {
            debugLog("local-asr: launch failed: \(error)")
        }
    }

    func stop() {
        guard let p = process else { return }
        if p.isRunning {
            debugLog("local-asr: stopping daemon (pid: \(p.processIdentifier))")
            p.terminate()
        }
        process = nil
    }
}
