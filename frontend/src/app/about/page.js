import Link from "next/link";

export default function About() {
  const team = [
    {
      name: "Praful Bhatt",
      role: "Full Stack Developer",
      bio: "Full Stack expert with a focus on AI integration and scalable systems.",
      avatar: "PB",
    },
    {
      name: "Prakash Chaudhary",
      role: "Back End Developer",
      bio: "Back End expert with a focus on scalable systems and API development.",
      avatar: "PC",
    },
    {
      name: "Madhulika Yadav",
      role: "Front End Developer",
      bio: "Front End expert with a focus on intuitive UI and UX design.",
      avatar: "MY",
    },
    {
      name: "Prisha Nepal",
      role: "Front End Developer",
      bio: "Front End expert with a focus on responsive web designs.",
      avatar: "PN",
    },
    {
      name: "Shristi Tamang",
      role: "Big data and AI specialist",
      bio: "Big Data expert with a focus on AI integration and scalable systems.",
      avatar: "ST",
    },
  ];

  return (
    <div>
      <section className="py-20 md:py-28">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
              About Humanizer
            </h1>
            <p className="text-xl text-gray-500">
              We are on a mission to help content creators maintain authenticity
              in an AI-driven world. Our tools bridge the gap between artificial
              and human writing.
            </p>
          </div>
        </div>
      </section>

      <section className="py-20 bg-gray-50">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-6">
                Our Story
              </h2>
              <div className="space-y-4 text-gray-600">
                <p>
                  Humanizer was born in 2023 from a simple observation: AI
                  writing tools were becoming incredibly powerful, but they left
                  a distinctive fingerprint that detection systems could easily
                  identify.
                </p>
                <p>
                  Our founders, a team of AI researchers and content creators,
                  saw the need for a tool that could help people use AI
                  responsibly while maintaining the authenticity of their voice.
                </p>
                <p>
                  Today, we serve over 50,000 users worldwide from students and
                  bloggers to marketing professionals and academic researchers.
                  Our technology has processed millions of texts, continuously
                  learning and improving.
                </p>
                <p>
                  We believe AI should be a tool that enhances human creativity,
                  not replaces it. Our mission is to help you leverage AI while
                  keeping your unique voice intact.
                </p>
              </div>
            </div>

            <div className="relative">
              <div className="bg-white border border-gray-200 rounded-2xl p-8 shadow-lg">
                <div className="grid grid-cols-2 gap-6">
                  <div className="text-center p-6 bg-gray-50 rounded-xl">
                    <p className="text-3xl font-bold text-gray-900">2023</p>
                    <p className="text-sm text-gray-500 mt-1">Founded</p>
                  </div>
                  <div className="text-center p-6 bg-gray-50 rounded-xl">
                    <p className="text-3xl font-bold text-gray-900">50K+</p>
                    <p className="text-sm text-gray-500 mt-1">Users</p>
                  </div>
                  <div className="text-center p-6 bg-gray-50 rounded-xl">
                    <p className="text-3xl font-bold text-gray-900">15</p>
                    <p className="text-sm text-gray-500 mt-1">Team Members</p>
                  </div>
                  <div className="text-center p-6 bg-gray-50 rounded-xl">
                    <p className="text-3xl font-bold text-gray-900">100+</p>
                    <p className="text-sm text-gray-500 mt-1">Countries</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Our Values
            </h2>
            <p className="text-lg text-gray-500 max-w-2xl mx-auto">
              The principles that guide everything we do
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="flex gap-5 p-6 bg-white border border-gray-200 rounded-2xl">
              <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center shrink-0 text-gray-700">
                <svg
                  className="w-6 h-6"
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
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Privacy First
                </h3>
                <p className="text-gray-500">
                  We never store your content. All processing happens in
                  real-time and data is immediately deleted.
                </p>
              </div>
            </div>

            <div className="flex gap-5 p-6 bg-white border border-gray-200 rounded-2xl">
              <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center shrink-0 text-gray-700">
                <svg
                  className="w-6 h-6"
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
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Always Improving
                </h3>
                <p className="text-gray-500">
                  Our algorithms are constantly updated to stay ahead of the
                  latest AI detection methods.
                </p>
              </div>
            </div>

            <div className="flex gap-5 p-6 bg-white border border-gray-200 rounded-2xl">
              <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center shrink-0 text-gray-700">
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                  />
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  User Focused
                </h3>
                <p className="text-gray-500">
                  Built by content creators for content creators. We understand
                  your needs because we share them.
                </p>
              </div>
            </div>

            <div className="flex gap-5 p-6 bg-white border border-gray-200 rounded-2xl">
              <div className="w-14 h-14 bg-gray-100 rounded-xl flex items-center justify-center shrink-0 text-gray-700">
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Global Reach
                </h3>
                <p className="text-gray-500">
                  Supporting 100+ languages to help content creators worldwide
                  communicate authentically.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-20 bg-gray-50">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Meet Our Team
            </h2>
            <p className="text-lg text-gray-500 max-w-2xl mx-auto">
              The passionate people behind Humanizer
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            {team.map((member, index) => (
              <div
                key={index}
                className="bg-white border border-gray-200 rounded-2xl p-6 text-center"
              >
                <div className="w-20 h-20 bg-gray-900 rounded-full flex items-center justify-center mx-auto mb-4">
                  <span className="text-xl font-bold text-white">
                    {member.avatar}
                  </span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900">
                  {member.name}
                </h3>
                <p className="text-sm text-gray-500 mb-3">{member.role}</p>
                <p className="text-sm text-gray-600">{member.bio}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="py-20 bg-gray-900">
        <div className="max-w-6xl mx-auto px-6">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-6">
              Our Mission
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              To empower content creators with AI tools that enhance human
              creativity rather than replace it, ensuring authenticity in every
              word written.
            </p>
          </div>
        </div>
      </section>

      <section className="py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center max-w-2xl mx-auto">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              Join Our Community
            </h2>
            <p className="text-lg text-gray-500 mb-8">
              Be part of the movement towards authentic AI-assisted writing
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                href="/humanizer"
                className="w-full sm:w-auto px-8 py-4 bg-gray-900 text-white font-semibold rounded-xl hover:bg-gray-700"
              >
                Try Humanizer Free
              </Link>
              <a
                href="#"
                className="w-full sm:w-auto px-8 py-4 border-2 border-gray-200 text-gray-700 font-semibold rounded-xl hover:border-gray-300 hover:bg-gray-50"
              >
                Contact Us
              </a>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
