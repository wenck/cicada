import Foundation

/// Keeps the local SenseVoice ASR daemon (`tools/local_asr_service.py` under `uv run`) running
/// while the speech settings point at a local address, and registers this app as an owner. The
/// script's own watchdog exits it once every owner PID is gone, so no quit path of this app — a
/// clean quit, SIGTERM, kill -9 or a crash — can orphan it.
@MainActor final class LocalASRService {
    static let shared = LocalASRService()
    /// The port this app is registered on as an owner; nil when it holds none.
    private var registeredPort: Int?
    /// A newer sync() supersedes answers still coming back for an older one.
    private var generation = 0
    /// The daemon takes seconds to load the model: a sync in that window must not launch another.
    private var launchedAt = Date.distantPast

    private let modelDir = NSString(string: "~/.cache/asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17").expandingTildeInPath
    private let cacheScript = NSString(string: "~/.cache/asr_models/local_asr_service.py").expandingTildeInPath
    private static let uvCandidates = [
        "/opt/homebrew/bin/uv", "/usr/local/bin/uv",
        NSString(string: "~/.local/bin/uv").expandingTildeInPath,
        NSString(string: "~/.cargo/bin/uv").expandingTildeInPath,
    ]

    /// The bundled script; the ~/.cache copy stays as the fallback for builds without it.
    private var scriptPath: String? {
        if let url = Bundle.main.url(forResource: "local_asr_service", withExtension: "py") { return url.path }
        return FileManager.default.fileExists(atPath: cacheScript) ? cacheScript : nil
    }

    private var uvPath: String? {
        Self.uvCandidates.first(where: { FileManager.default.fileExists(atPath: $0) })
    }

    /// Settings say where hearing goes: the local daemon is wanted only while the hear address is
    /// a port on this Mac.
    func sync(_ c: AIConfig) {
        generation += 1
        let gen = generation
        guard let port = wantedPort(c) else {
            unregister()
            return
        }
        if let registered = registeredPort, registered != port { unregister() }
        Task {
            switch await probe(port) {
            case .ours:
                // A service already up (started by another copy or before this launch): adopt it.
                guard gen == generation, registeredPort != port else { return }
                await ownerRequest("POST", port: port)
                guard gen == generation else { return }
                registeredPort = port
                debugLog("local-asr: adopted the service on port \(port)")
            case .foreign:
                debugLog("local-asr: port \(port) is taken by something else")
            case .none:
                guard gen == generation,
                      !(registeredPort == port && Date().timeIntervalSince(launchedAt) < 30) else { return }
                launch(port: port)
            }
        }
    }

    /// The port of a wanted service, or nil with the reason logged.
    private func wantedPort(_ c: AIConfig) -> Int? {
        guard let base = c.hearBase, let host = base.host, AIProvider.isLocal(host), let port = base.port
        else { return nil }
        guard FileManager.default.fileExists(atPath: (modelDir as NSString).appendingPathComponent("model.int8.onnx")) else {
            debugLog("local-asr: model not found in \(modelDir)")
            return nil
        }
        guard scriptPath != nil else {
            debugLog("local-asr: script not found (bundle Resources or \(cacheScript))")
            return nil
        }
        guard uvPath != nil else {
            debugLog("local-asr: uv not found")
            return nil
        }
        return port
    }

    /// What answers on `port`: our service, a foreign one, or nothing.
    private enum Probe { case ours, foreign, none }

    private func probe(_ port: Int) async -> Probe {
        guard let url = URL(string: "http://127.0.0.1:\(port)/health") else { return .none }
        var req = URLRequest(url: url)
        req.timeoutInterval = 1
        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard (resp as? HTTPURLResponse) != nil else { return .none }
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            return json?["service"] as? String == "cicada-local-asr" ? .ours : .foreign
        } catch {
            return .none
        }
    }

    private func unregister() {
        guard let port = registeredPort else { return }
        registeredPort = nil
        Task { await ownerRequest("DELETE", port: port) }
    }

    private func ownerRequest(_ method: String, port: Int) async {
        guard let url = URL(string: "http://127.0.0.1:\(port)/owner?pid=\(getpid())") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.timeoutInterval = 2
        _ = try? await URLSession.shared.data(for: req)
    }

    private func launch(port: Int) {
        guard let uv = uvPath, let script = scriptPath else { return }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: uv)
        p.arguments = ["run", script]
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + (env["PATH"] ?? "")
        env["CICADA_ASR_PORT"] = "\(port)"
        env["CICADA_OWNER_PID"] = "\(getpid())"
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
            registeredPort = port
            launchedAt = Date()
            debugLog("local-asr: launched daemon on port \(port) (pid: \(p.processIdentifier))")
        } catch {
            debugLog("local-asr: launch failed: \(error)")
        }
    }
}
