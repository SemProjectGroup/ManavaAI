"use client";

import { useState } from "react";
import ProtectedRoute from "../../components/ProtectedRoute";
import { useAuth } from "../../context/AuthContext";

function HumanizerContent() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");
  const [activeTab, setActiveTab] = useState("humanize");
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);
  const [copied, setCopied] = useState(false);
  const { user } = useAuth();

  const handleDetect = async () => {
    if (!inputText.trim()) return;

    setIsProcessing(true);
    setDetectionResult(null);
    setOutputText("");

    await new Promise((resolve) => setTimeout(resolve, 1500));

    const aiScore = Math.floor(Math.random() * 40) + 60;
    setDetectionResult({
      aiScore: aiScore,
      humanScore: 100 - aiScore,
      verdict: aiScore > 70 ? "Likely AI Generated" : "Mixed Content",
      details: [
        { label: "Sentence uniformity", flag: true },
        { label: "Vocabulary patterns", flag: aiScore > 75 },
        { label: "Structural repetition", flag: aiScore > 65 },
        { label: "Natural flow", flag: false },
      ],
    });

    setIsProcessing(false);
  };

  const handleHumanize = async () => {
    if (!inputText.trim()) return;

    setIsProcessing(true);
    setDetectionResult(null);
    setOutputText("");

    await new Promise((resolve) => setTimeout(resolve, 2000));

    setOutputText(
      `Here is the humanized version of your text:\n\n"${inputText.substring(
        0,
        100
      )}${
        inputText.length > 100 ? "..." : ""
      }"\n\nThe text has been transformed to sound more natural and human-like. Key changes include varied sentence structures, natural transitions, and conversational tone adjustments.\n\nAI Detection Score: Reduced from 85% to 12%\nReadability: Improved\nOriginal meaning: Preserved`
    );

    setIsProcessing(false);
  };

  const handleCopy = async () => {
    if (!outputText) return;
    await navigator.clipboard.writeText(outputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleClear = () => {
    setInputText("");
    setOutputText("");
    setDetectionResult(null);
  };

  const handleProcess = () => {
    if (activeTab === "detect") {
      handleDetect();
    } else {
      handleHumanize();
    }
  };

  return (
    <div>
      <section className="py-12 md:py-16">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-2xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-green-50 border border-green-200 rounded-full text-sm text-green-700 mb-4">
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
              Signed in as {user?.email}
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
              AI Humanizer Tool
            </h1>
            <p className="text-lg text-gray-500">
              Detect AI-generated content or transform it into natural human
              writing
            </p>
          </div>
        </div>
      </section>

      <section className="pb-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="bg-white border border-gray-200 rounded-2xl shadow-sm overflow-hidden">
            <div className="border-b border-gray-200 bg-gray-50">
              <div className="flex">
                <button
                  onClick={() => {
                    setActiveTab("humanize");
                    setDetectionResult(null);
                    setOutputText("");
                  }}
                  className={`flex-1 px-6 py-4 text-sm font-medium ${
                    activeTab === "humanize"
                      ? "text-gray-900 bg-white border-b-2 border-gray-900"
                      : "text-gray-500 hover:text-gray-700"
                  }`}
                >
                  <div className="flex items-center justify-center gap-2">
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                      />
                    </svg>
                    Humanize Text
                  </div>
                </button>
                <button
                  onClick={() => {
                    setActiveTab("detect");
                    setDetectionResult(null);
                    setOutputText("");
                  }}
                  className={`flex-1 px-6 py-4 text-sm font-medium ${
                    activeTab === "detect"
                      ? "text-gray-900 bg-white border-b-2 border-gray-900"
                      : "text-gray-500 hover:text-gray-700"
                  }`}
                >
                  <div className="flex items-center justify-center gap-2">
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                      />
                    </svg>
                    Detect AI
                  </div>
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-gray-200">
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <label className="text-sm font-medium text-gray-700">
                    Input Text
                  </label>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-gray-400">
                      {inputText.length} / 5000
                    </span>
                    <button
                      onClick={handleClear}
                      className="text-xs text-gray-500 hover:text-gray-700"
                    >
                      Clear
                    </button>
                  </div>
                </div>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value.slice(0, 5000))}
                  placeholder={
                    activeTab === "detect"
                      ? "Paste text here to check if it was written by AI..."
                      : "Paste your AI-generated text here to humanize it..."
                  }
                  className="w-full h-80 p-4 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 resize-none focus:border-gray-300 focus:bg-white"
                />
              </div>

              <div className="p-6 bg-gray-50">
                <div className="flex items-center justify-between mb-4">
                  <label className="text-sm font-medium text-gray-700">
                    {activeTab === "detect"
                      ? "Detection Results"
                      : "Humanized Output"}
                  </label>
                  {outputText && (
                    <button
                      onClick={handleCopy}
                      className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700"
                    >
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                        />
                      </svg>
                      {copied ? "Copied!" : "Copy"}
                    </button>
                  )}
                </div>

                <div className="w-full h-80 p-4 bg-white border border-gray-200 rounded-xl overflow-auto">
                  {isProcessing ? (
                    <div className="h-full flex flex-col items-center justify-center gap-4">
                      <div className="w-8 h-8 border-2 border-gray-200 border-t-gray-900 rounded-full animate-spin"></div>
                      <span className="text-sm text-gray-500">
                        {activeTab === "detect"
                          ? "Analyzing text..."
                          : "Humanizing your text..."}
                      </span>
                    </div>
                  ) : detectionResult ? (
                    <div className="space-y-6">
                      <div className="flex items-center justify-center gap-8">
                        <div className="text-center">
                          <div className="relative w-24 h-24">
                            <svg className="w-24 h-24 transform -rotate-90">
                              <circle
                                cx="48"
                                cy="48"
                                r="40"
                                stroke="#e5e7eb"
                                strokeWidth="8"
                                fill="none"
                              />
                              <circle
                                cx="48"
                                cy="48"
                                r="40"
                                stroke="#111827"
                                strokeWidth="8"
                                fill="none"
                                strokeDasharray={`${
                                  detectionResult.aiScore * 2.51
                                } 251`}
                              />
                            </svg>
                            <div className="absolute inset-0 flex items-center justify-center">
                              <span className="text-2xl font-bold text-gray-900">
                                {detectionResult.aiScore}%
                              </span>
                            </div>
                          </div>
                          <p className="text-sm text-gray-500 mt-2">
                            AI Probability
                          </p>
                        </div>
                      </div>

                      <div className="text-center">
                        <span
                          className={`inline-flex px-4 py-2 rounded-full text-sm font-medium ${
                            detectionResult.aiScore > 70
                              ? "bg-gray-900 text-white"
                              : "bg-gray-200 text-gray-700"
                          }`}
                        >
                          {detectionResult.verdict}
                        </span>
                      </div>

                      <div className="space-y-3">
                        <p className="text-sm font-medium text-gray-700">
                          Analysis Details:
                        </p>
                        {detectionResult.details.map((detail, index) => (
                          <div
                            key={index}
                            className="flex items-center justify-between text-sm"
                          >
                            <span className="text-gray-600">
                              {detail.label}
                            </span>
                            <span
                              className={
                                detail.flag ? "text-gray-900" : "text-gray-400"
                              }
                            >
                              {detail.flag ? "Detected" : "Natural"}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : outputText ? (
                    <p className="text-gray-900 whitespace-pre-wrap">
                      {outputText}
                    </p>
                  ) : (
                    <div className="h-full flex items-center justify-center">
                      <p className="text-gray-400 text-center">
                        {activeTab === "detect"
                          ? "Detection results will appear here..."
                          : "Humanized text will appear here..."}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="p-6 bg-white border-t border-gray-200">
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <button
                  onClick={handleProcess}
                  disabled={!inputText.trim() || isProcessing}
                  className="w-full sm:w-auto px-10 py-4 bg-gray-900 text-white font-semibold rounded-xl hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing
                    ? "Processing..."
                    : activeTab === "detect"
                    ? "Detect AI Content"
                    : "Humanize Text"}
                </button>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
            <div className="p-6 bg-white border border-gray-200 rounded-xl">
              <div className="w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-gray-700"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                  />
                </svg>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">100% Private</h3>
              <p className="text-sm text-gray-500">
                Your text is never stored. All processing happens in real-time
                and is immediately deleted.
              </p>
            </div>

            <div className="p-6 bg-white border border-gray-200 rounded-xl">
              <div className="w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-gray-700"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">
                Instant Results
              </h3>
              <p className="text-sm text-gray-500">
                Get detection results or humanized text in seconds. No waiting,
                no queues.
              </p>
            </div>

            <div className="p-6 bg-white border border-gray-200 rounded-xl">
              <div className="w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center mb-4">
                <svg
                  className="w-6 h-6 text-gray-700"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                  />
                </svg>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">
                ??% Bypass Rate
              </h3>
              <p className="text-sm text-gray-500">
                Our humanized content bypasses GPTZero, Originality.ai,
                Turnitin, and more.
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default function Humanizer() {
  return (
    <ProtectedRoute>
      <HumanizerContent />
    </ProtectedRoute>
  );
}
