//
//  TranscriptionEngineTests.swift
//  PindropTests
//
//  Created on 2026-01-30.
//

import AVFoundation
import Foundation
import Speech
import Testing
@testable import Pindrop

@MainActor
@Suite
struct TranscriptionEngineTests {
    @Test func transcriptionEngineStateEquatable() {
        #expect(TranscriptionEngineState.unloaded == .unloaded)
        #expect(TranscriptionEngineState.loading == .loading)
        #expect(TranscriptionEngineState.ready == .ready)
        #expect(TranscriptionEngineState.transcribing == .transcribing)
        #expect(TranscriptionEngineState.error == .error)

        #expect(TranscriptionEngineState.unloaded != .loading)
        #expect(TranscriptionEngineState.ready != .transcribing)
    }

    @Test func transcriptionEngineStateCases() {
        let states: [TranscriptionEngineState] = [.unloaded, .loading, .ready, .transcribing, .error]
        #expect(states.count == 5)
    }

    @Test func mockEngineConformsToProtocol() {
        let engine = MockTranscriptionEngine()
        #expect(engine is TranscriptionEngine)
    }

    @Test func mockEngineInitialState() {
        let engine = MockTranscriptionEngine()
        #expect(engine.state == .unloaded)
    }

    @Test func mockEngineStateTransitions() async throws {
        let engine = MockTranscriptionEngine()

        #expect(engine.state == .unloaded)

        try await engine.loadModel(name: "tiny", downloadBase: nil)
        #expect(engine.state == .ready)

        let audioData = Data([0x00, 0x01, 0x02, 0x03])
        _ = try await engine.transcribe(audioData: audioData)
        #expect(engine.state == .ready)

        await engine.unloadModel()
        #expect(engine.state == .unloaded)
    }

    @Test func mockEngineLoadByPath() async throws {
        let engine = MockTranscriptionEngine()
        try await engine.loadModel(path: "/path/to/model")
        #expect(engine.state == .ready)
    }

    @Test func mockEngineTranscription() async throws {
        let engine = MockTranscriptionEngine()
        try await engine.loadModel(name: "tiny", downloadBase: nil)

        let audioData = Data([0x00, 0x01, 0x02, 0x03])
        let result = try await engine.transcribe(audioData: audioData)

        #expect(result == "Mock transcription result")
    }

    @Test func mockEngineErrorState() async {
        let engine = MockTranscriptionEngine()
        engine.shouldFailLoad = true

        do {
            try await engine.loadModel(name: "tiny", downloadBase: nil)
            Issue.record("Expected load failure")
        } catch {
            #expect(engine.state == .error)
        }
    }

    @Test func openAIEngineUploadsMultipartAudioAndParsesTranscript() async throws {
        let session = OpenAITranscriptionSessionStub()
        session.responseData = Data(#"{"text":"Cloud transcript"}"#.utf8)
        session.statusCode = 200
        let engine = OpenAITranscriptionEngine(
            apiKeyProvider: { "sk-test-key" },
            session: session
        )

        try await engine.loadModel(name: "openai_gpt-4o-transcribe", downloadBase: nil)
        let audioData = Data(count: 16_000 * MemoryLayout<Float>.size)
        let transcript = try await engine.transcribe(
            audioData: audioData,
            options: TranscriptionOptions(
                language: .english,
                vocabularyBiasWords: ["Pindrop", "WhisperKit"]
            )
        )

        #expect(transcript == "Cloud transcript")
        let request = try #require(session.lastRequest)
        #expect(request.url?.absoluteString == "https://api.openai.com/v1/audio/transcriptions")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer sk-test-key")
        let contentType = try #require(request.value(forHTTPHeaderField: "Content-Type"))
        #expect(contentType.hasPrefix("multipart/form-data; boundary="))
        let body = String(decoding: try #require(request.httpBody), as: UTF8.self)
        #expect(body.contains("name=\"model\""))
        #expect(body.contains("gpt-4o-transcribe"))
        #expect(body.contains("name=\"language\""))
        #expect(body.contains("\r\n\r\nen\r\n"))
        #expect(body.contains("name=\"prompt\""))
        #expect(body.contains("Pindrop, WhisperKit"))
        #expect(body.contains("filename=\"audio.m4a\""))
    }

    @Test func openAIEngineRequiresConfiguredAPIKey() async {
        let engine = OpenAITranscriptionEngine(apiKeyProvider: { "" })

        await #expect(throws: OpenAITranscriptionEngine.EngineError.self) {
            try await engine.loadModel(name: "openai_gpt-4o-transcribe", downloadBase: nil)
        }
        #expect(engine.state == .error)
    }

    @Test func openAIEngineSurfacesAPIErrorMessage() async throws {
        let session = OpenAITranscriptionSessionStub()
        session.responseData = Data(#"{"error":{"message":"Incorrect API key provided"}}"#.utf8)
        session.statusCode = 401
        let engine = OpenAITranscriptionEngine(
            apiKeyProvider: { "sk-invalid" },
            session: session
        )

        try await engine.loadModel(name: "openai_gpt-4o-mini-transcribe", downloadBase: nil)
        await #expect(throws: OpenAITranscriptionEngine.EngineError.self) {
            _ = try await engine.transcribe(
                audioData: Data(count: 4),
                options: TranscriptionOptions()
            )
        }
        #expect(engine.error?.localizedDescription.contains("Incorrect API key provided") == true)
        #expect(engine.state == .ready)
    }
}

@MainActor
final class MockTranscriptionEngine: TranscriptionEngine {
    private(set) var state: TranscriptionEngineState = .unloaded
    var shouldFailLoad = false
    var mockTranscriptionResult = "Mock transcription result"
    private(set) var lastOptions: TranscriptionOptions?

    func loadModel(path: String) async throws {
        if shouldFailLoad {
            state = .error
            throw MockError.loadFailed
        }
        state = .ready
    }

    func loadModel(name: String, downloadBase: URL?) async throws {
        if shouldFailLoad {
            state = .error
            throw MockError.loadFailed
        }
        state = .ready
    }

    func transcribe(audioData: Data, options: TranscriptionOptions) async throws -> String {
        guard state == .ready else {
            throw MockError.modelNotLoaded
        }

        lastOptions = options
        state = .transcribing
        try await Task.sleep(nanoseconds: 10_000_000)
        state = .ready
        return mockTranscriptionResult
    }

    func unloadModel() async {
        state = .unloaded
    }
}


@MainActor
private final class OpenAITranscriptionSessionStub: URLSessionProtocol {
    var responseData = Data()
    var statusCode = 200
    private(set) var lastRequest: URLRequest?

    func data(for request: URLRequest) async throws -> (Data, URLResponse) {
        lastRequest = request
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: statusCode,
            httpVersion: nil,
            headerFields: ["x-request-id": "req-test"]
        )!
        return (responseData, response)
    }
}

enum MockError: Error {
    case loadFailed
    case modelNotLoaded
}

// MARK: - Apple Speech utterances

/// Issue #85: on-device recognition starts a new transcription after each pause.
/// The result sequences below are the ones `SFSpeechRecognizer` reported for
/// synthesized speech with 3 s pauses (start and end are audio times in seconds).
@Suite("AppleSpeechUtteranceAccumulator")
struct AppleSpeechUtteranceAccumulatorTests {
    @Test func isEmptyWithoutResults() {
        #expect(AppleSpeechUtteranceAccumulator().text == "")
    }

    @Test func keepsEveryUtteranceBeforeAPause() {
        var sut = AppleSpeechUtteranceAccumulator()

        sut.add(text: "I went to the store", start: 0.0, end: 2.58)
        sut.add(text: "Then I walked home", start: 5.64, end: 8.43)
        sut.add(text: "We talked about the project", start: 11.49, end: 14.55)

        #expect(sut.text == "I went to the store Then I walked home We talked about the project")
    }

    @Test func doesNotRepeatTheLastUtteranceWhenTheFinalResultRepeatsIt() {
        var sut = AppleSpeechUtteranceAccumulator()

        // With silence at the end, the final result repeats the last utterance.
        sut.add(text: "I went to the store", start: 0.0, end: 2.58)
        sut.add(text: "Then I walked home", start: 5.64, end: 7.53)
        sut.add(text: "Then I walked home", start: 5.64, end: 7.53)

        #expect(sut.text == "I went to the store Then I walked home")
    }

    @Test func keepsTheSamePhraseWhenItIsSpokenTwice() {
        var sut = AppleSpeechUtteranceAccumulator()

        sut.add(text: "Yes", start: 0.0, end: 0.4)
        sut.add(text: "Yes", start: 3.5, end: 3.9)

        #expect(sut.text == "Yes Yes")
    }

    @Test func aSingleFinalResultIsReturnedUnchanged() {
        var sut = AppleSpeechUtteranceAccumulator()

        sut.add(text: "I went to the store then I walked home", start: 0.0, end: 5.46)

        #expect(sut.text == "I went to the store then I walked home")
    }

    @Test func aNewerVersionReplacesTheUtterancesItOverlaps() {
        var sut = AppleSpeechUtteranceAccumulator()

        // Partial results grow one utterance.
        sut.add(text: "I went", start: 0.0, end: 0.6)
        sut.add(text: "I went to the store", start: 0.0, end: 2.58)
        sut.add(text: "Then I", start: 5.64, end: 6.0)
        #expect(sut.text == "I went to the store Then I")

        // A result with the full text replaces all utterances that it covers.
        sut.add(text: "I went to the store then I walked home", start: 0.0, end: 8.43)
        #expect(sut.text == "I went to the store then I walked home")
    }

    @Test func ignoresEmptyResults() {
        var sut = AppleSpeechUtteranceAccumulator()

        sut.add(text: "I went to the store", start: 0.0, end: 2.58)
        sut.add(text: "  ", start: 5.0, end: 5.0)
        sut.add(text: "", start: nil, end: nil)

        #expect(sut.text == "I went to the store")
    }

    @Test func aResultWithoutWordTimingFollowsTheTimedUtterances() {
        var sut = AppleSpeechUtteranceAccumulator()

        sut.add(text: "I went to the store", start: 0.0, end: 2.58)
        sut.add(text: "Then I walked home", start: nil, end: nil)

        #expect(sut.text == "I went to the store Then I walked home")
    }

    @Test func untimedResultsKeepOnlyTheNewestText() {
        var sut = AppleSpeechUtteranceAccumulator()

        // Without word timing a new utterance and a newer version look the same.
        sut.add(text: "I went", start: nil, end: nil)
        sut.add(text: "I went to the store", start: nil, end: nil)

        #expect(sut.text == "I went to the store")
    }
}

// MARK: - Integration (real on-device recognition) — opt-in only

/// Needs the on-device speech model for en-US. Run with:
///   TEST_RUNNER_PINDROP_RUN_APPLE_SPEECH_TESTS=1 xcodebuild test ... \
///     -only-testing:PindropTests/AppleSpeechEngineIntegrationTests
@MainActor
@Suite(
    "AppleSpeechEngine (integration, on-device recognition)",
    .enabled(
        if: ProcessInfo.processInfo.environment["PINDROP_RUN_APPLE_SPEECH_TESTS"] == "1",
        "Apple Speech recognition tests are disabled by default. Run with PINDROP_RUN_APPLE_SPEECH_TESTS=1."
    )
)
struct AppleSpeechEngineIntegrationTests {
    @Test func transcriptKeepsTheSpeechBeforeEachPause() async throws {
        let recognizer = try #require(SFSpeechRecognizer(locale: Locale(identifier: "en-US")))
        try #require(recognizer.isAvailable && recognizer.supportsOnDeviceRecognition)
        let buffer = try SpeechSynthesisTestSupport.synthesizeSpeechBuffer(
            "I went to the store this morning and bought some apples. [[slnc 3000]] "
                + "Then I walked home through the park and called my friend. [[slnc 3000]] "
                + "We talked about the new project that starts next week. [[slnc 4000]]"
        )

        let transcript = try await AppleSpeechEngine()
            .performRecognition(using: recognizer, buffer: buffer)
            .lowercased()

        for keyword in ["apples", "park", "project"] {
            #expect(transcript.contains(keyword), "Missing '\(keyword)' in: '\(transcript)'")
        }
        #expect(
            transcript.components(separatedBy: "project").count == 2,
            "The last utterance must appear once: '\(transcript)'"
        )
    }
}
