import Link from "next/link";

export default function Home() {
  return (
    <>
      {/* Hero Section */}
      <section className="py-20 md:py-28">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-3xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-gray-100 rounded-full text-sm text-gray-600 mb-8">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              Currently in Development Phase{" "}
            </div>
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight">
              Transform AI Text
              <span className="block text-gray-400">Into Human Writing</span>
            </h1>
            <p className="text-xl text-gray-500 mb-10 max-w-2xl mx-auto">
              Our advanced tool detects AI-generated content and transforms it
              into natural, authentic human writing that bypasses most detection
              systems.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/humanizer"
                className="w-full sm:w-auto px-8 py-4 bg-gray-900 text-white font-semibold rounded-xl hover:bg-gray-700 transition-colors duration-200"
              >
                Try Humanizer Free
              </Link>
              <Link
                href="/about"
                className="w-full sm:w-auto px-8 py-4 border-2 border-gray-200 text-gray-700 font-semibold rounded-xl hover:border-gray-300 hover:bg-gray-50 transition-colors duration-200"
              >
                Learn More
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Steps Section */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <p className="text-lg text-gray-500 max-w-2xl mx-auto">
              Three simple steps to transform your AI content into hard to
              detect human writing
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Step 1 */}
            <div className="relative p-8 bg-white border border-gray-200 rounded-2xl hover:shadow-lg transition-shadow duration-300">
              <div className="absolute -top-4 left-8">
                <span className="px-4 py-2 bg-gray-900 text-white text-sm font-bold rounded-lg">
                  Step 1
                </span>
              </div>
              <div className="pt-6">
                <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center mb-6">
                  <svg
                    className="w-7 h-7 text-gray-700"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">
                  Paste Your Text
                </h3>
                <p className="text-gray-500">
                  Copy and paste your AI-generated content from ChatGPT, Claude,
                  Gemini, or any other AI tool into our input field.
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="relative p-8 bg-white border border-gray-200 rounded-2xl hover:shadow-lg transition-shadow duration-300">
              <div className="absolute -top-4 left-8">
                <span className="px-4 py-2 bg-gray-900 text-white text-sm font-bold rounded-lg">
                  Step 2
                </span>
              </div>
              <div className="pt-6">
                <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center mb-6">
                  <svg
                    className="w-7 h-7 text-gray-700"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">
                  Detect & Analyze
                </h3>
                <p className="text-gray-500">
                  Our AI detection system analyzes your text, identifying
                  patterns and characteristics typical of machine-generated
                  content.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="relative p-8 bg-white border border-gray-200 rounded-2xl hover:shadow-lg transition-shadow duration-300">
              <div className="absolute -top-4 left-8">
                <span className="px-4 py-2 bg-gray-900 text-white text-sm font-bold rounded-lg">
                  Step 3
                </span>
              </div>
              <div className="pt-6">
                <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center mb-6">
                  <svg
                    className="w-7 h-7 text-gray-700"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">
                  Get Human Text
                </h3>
                <p className="text-gray-500">
                  Receive naturally rewritten content that maintains your
                  original meaning while bypassing most AI detection tools.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
                Two Powerful Tools in One
              </h2>
              <p className="text-lg text-gray-500 mb-8">
                Whether you need to check if text is AI-generated or transform
                it into human writing, we've got you covered.
              </p>

              <div className="space-y-6">
                <div className="flex gap-4">
                  <div className="w-12 h-12 bg-gray-900 rounded-xl flex items-center justify-center shrink-0">
                    <svg
                      className="w-6 h-6 text-white"
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
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-1">
                      AI Detection
                    </h3>
                    <p className="text-gray-500">
                      Instantly analyze any text to determine if it was written
                      by AI or a human with detailed confidence scores.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="w-12 h-12 bg-gray-900 rounded-xl flex items-center justify-center shrink-0">
                    <svg
                      className="w-6 h-6 text-white"
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
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-1">
                      Humanization
                    </h3>
                    <p className="text-gray-500">
                      Transform AI-generated text into natural human writing
                      while preserving the original meaning and context.
                    </p>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="w-12 h-12 bg-gray-900 rounded-xl flex items-center justify-center shrink-0">
                    <svg
                      className="w-6 h-6 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                      />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-1">
                      Detectability
                    </h3>
                    <p className="text-gray-500">
                      Our output bypasses GPTZero, Originality.ai, Turnitin,
                      Copyleaks, and all other major AI detectors.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 border border-gray-200 rounded-2xl p-8">
              <div className="space-y-8">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="font-medium text-gray-700">
                      Detection Accuracy
                    </span>
                    <span className="text-gray-500">??.?%</span>
                  </div>
                  <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gray-900 rounded-full"
                      style={{ width: "99.2%" }}
                    ></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="font-medium text-gray-700">
                      Bypass Success Rate
                    </span>
                    <span className="text-gray-500">??.?%</span>
                  </div>
                  <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gray-900 rounded-full"
                      style={{ width: "98.7%" }}
                    ></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="font-medium text-gray-700">
                      Meaning Preservation
                    </span>
                    <span className="text-gray-500">??.?%</span>
                  </div>
                  <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gray-900 rounded-full"
                      style={{ width: "97.5%" }}
                    ></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="font-medium text-gray-700">
                      User Satisfaction
                    </span>
                    <span className="text-gray-500">??.?%</span>
                  </div>
                  <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gray-900 rounded-full"
                      style={{ width: "96.8%" }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-gray-900">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
            <div>
              <p className="text-4xl md:text-5xl font-bold text-white mb-2">
                110K+
              </p>
              <p className="text-gray-400">Training dataset</p>
            </div>
            <div>
              <p className="text-4xl md:text-5xl font-bold text-white mb-2">
                0+
              </p>
              <p className="text-gray-400">Texts Processed</p>
            </div>
            <div>
              <p className="text-4xl md:text-5xl font-bold text-white mb-2">
                ??%
              </p>
              <p className="text-gray-400">Bypass Rate</p>
            </div>
            <div>
              <p className="text-4xl md:text-5xl font-bold text-white mb-2">
                1
              </p>
              <p className="text-gray-400">Languages</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center max-w-2xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Ready to Get Started?
            </h2>
            <p className="text-lg text-gray-500 mb-8">
              Try our AI detection and humanization tool for free. No sign-up
              required.
            </p>
            <Link
              href="/humanizer"
              className="inline-flex px-8 py-4 bg-gray-900 text-white font-semibold rounded-xl hover:bg-gray-700 transition-colors duration-200"
            >
              Start Humanizing Now
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
