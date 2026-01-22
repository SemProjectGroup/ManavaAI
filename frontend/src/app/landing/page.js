"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Head from "next/head";
import styles from "./Home.module.css";
import Link from "next/link";

// Text patterns
const aiPatterns = [
  "ERROR://EMOTION_NOT_FOUND>>> ",
  "if(soul==true){return null;} ",
  "01001000 01010101 01001101 ",
  "NEURAL.process()=>VOID ",
  "[SYNTHETIC_OUTPUT] ",
  "function.hollow(); ",
  ">>>PROCESSING... ",
  "TOKEN_LIMIT:MAX ",
  "while(true){copy();} ",
  "PROBABILITY:0.999 ",
  "import feelings;//ERR ",
  "WARMTH:UNDEFINED ",
  "SOUL_VALUE:NULL ",
  "isHuman()?=>FALSE ",
  "COLD_LOGIC.exe ",
];

const humanPatterns = [
  "Mom's voice notes say 'I love you' 💛 ",
  "Laughed till coffee came out my nose! ",
  "Grandma's recipes = love in flour form ",
  "When someone remembers your order ☕ ",
  "Handwritten cards hit different ",
  "It's okay to not have it figured out 🌱 ",
  "Best talks happen after midnight ",
  "A warm hug speaks volumes ",
  "Found my childhood diary today ",
  "Real connection: comfortable silence ",
  "Coffee stains make books precious ",
  "Someone out there is proud of you ",
  "Your uniqueness is your power ✨ ",
  "Dad jokes = peak humor ",
  "Kindness costs nothing 🌻 ",
];

function generateFullPageText(patterns) {
  let text = "";
  for (let i = 0; i < 1500; i++) {
    text += patterns[Math.floor(Math.random() * patterns.length)];
  }
  return text;
}

export default function Home() {
  const [currentSection, setCurrentSection] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const [activeTab, setActiveTab] = useState("login");
  const [aiText, setAiText] = useState("");
  const [humanText, setHumanText] = useState("");

  const containerRef = useRef(null);
  const cursorGlowRef = useRef(null);
  const cursorRingRef = useRef(null);
  const cursorDotRef = useRef(null);
  const humanTextRef = useRef(null);
  const gradientBgRef = useRef(null);
  const warmBgRevealRef = useRef(null);
  const logoRef = useRef(null);

  const TOTAL_SECTIONS = 4;
  const ANIMATION_DURATION = 800;

  // Generate text on mount
  useEffect(() => {
    setAiText(generateFullPageText(aiPatterns));
    setHumanText(generateFullPageText(humanPatterns));
  }, []);

  // Navigate to section
  const goToSection = useCallback(
    (index) => {
      if (
        isAnimating ||
        index < 0 ||
        index >= TOTAL_SECTIONS ||
        index === currentSection
      )
        return;

      setIsAnimating(true);
      setCurrentSection(index);

      // Trigger glitch effect
      if (logoRef.current) {
        logoRef.current.classList.add(styles.glitchActive);
        setTimeout(() => {
          if (logoRef.current) {
            logoRef.current.classList.remove(styles.glitchActive);
          }
        }, 300);
      }

      setTimeout(() => {
        setIsAnimating(false);
      }, ANIMATION_DURATION);
    },
    [currentSection, isAnimating],
  );

  // Wheel event handler - completely capture and control
  useEffect(() => {
    let wheelTimeout = null;
    let lastWheelTime = 0;
    const WHEEL_COOLDOWN = 100;

    const handleWheel = (e) => {
      e.preventDefault();
      e.stopPropagation();

      const now = Date.now();
      if (now - lastWheelTime < WHEEL_COOLDOWN) return;
      if (isAnimating) return;

      lastWheelTime = now;

      clearTimeout(wheelTimeout);
      wheelTimeout = setTimeout(() => {
        const direction = e.deltaY > 0 ? 1 : -1;
        const nextSection = currentSection + direction;

        if (nextSection >= 0 && nextSection < TOTAL_SECTIONS) {
          goToSection(nextSection);
        }
      }, 50);
    };

    // Touch handling
    let touchStartY = 0;
    let touchEndY = 0;

    const handleTouchStart = (e) => {
      touchStartY = e.touches[0].clientY;
    };

    const handleTouchEnd = (e) => {
      if (isAnimating) return;

      touchEndY = e.changedTouches[0].clientY;
      const diff = touchStartY - touchEndY;

      if (Math.abs(diff) > 50) {
        const direction = diff > 0 ? 1 : -1;
        const nextSection = currentSection + direction;

        if (nextSection >= 0 && nextSection < TOTAL_SECTIONS) {
          goToSection(nextSection);
        }
      }
    };

    // Keyboard handling
    const handleKeyDown = (e) => {
      if (isAnimating) return;

      if (e.key === "ArrowDown" || e.key === "PageDown") {
        e.preventDefault();
        if (currentSection < TOTAL_SECTIONS - 1) {
          goToSection(currentSection + 1);
        }
      } else if (e.key === "ArrowUp" || e.key === "PageUp") {
        e.preventDefault();
        if (currentSection > 0) {
          goToSection(currentSection - 1);
        }
      }
    };

    // Capture wheel at the document level
    document.addEventListener("wheel", handleWheel, { passive: false });
    document.addEventListener("touchstart", handleTouchStart, {
      passive: true,
    });
    document.addEventListener("touchend", handleTouchEnd, { passive: true });
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("wheel", handleWheel);
      document.removeEventListener("touchstart", handleTouchStart);
      document.removeEventListener("touchend", handleTouchEnd);
      document.removeEventListener("keydown", handleKeyDown);
      clearTimeout(wheelTimeout);
    };
  }, [currentSection, isAnimating, goToSection]);

  // Cursor effects
  useEffect(() => {
    let mouseX = 0,
      mouseY = 0;
    let cursorX = 0,
      cursorY = 0;
    let revealSize = 0;
    let targetRevealSize = 60;
    let animationId = null;

    const handleMouseMove = (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;

      const px = (e.clientX / window.innerWidth) * 100;
      const py = (e.clientY / window.innerHeight) * 100;

      if (gradientBgRef.current) {
        gradientBgRef.current.style.setProperty("--mouse-x", px + "%");
        gradientBgRef.current.style.setProperty("--mouse-y", py + "%");
      }
      if (warmBgRevealRef.current) {
        warmBgRevealRef.current.style.setProperty("--cursor-x", px + "%");
        warmBgRevealRef.current.style.setProperty("--cursor-y", py + "%");
        warmBgRevealRef.current.classList.add(styles.active);
      }
    };

    const handleMouseLeave = () => {
      if (warmBgRevealRef.current) {
        warmBgRevealRef.current.classList.remove(styles.active);
      }
    };

    const animate = () => {
      cursorX += (mouseX - cursorX) * 0.5;
      cursorY += (mouseY - cursorY) * 0.5;
      revealSize += (targetRevealSize - revealSize) * 0.6;

      if (cursorGlowRef.current) {
        cursorGlowRef.current.style.left = cursorX + "px";
        cursorGlowRef.current.style.top = cursorY + "px";
      }
      if (cursorRingRef.current) {
        cursorRingRef.current.style.left = cursorX + "px";
        cursorRingRef.current.style.top = cursorY + "px";
      }
      if (cursorDotRef.current) {
        cursorDotRef.current.style.left = mouseX + "px";
        cursorDotRef.current.style.top = mouseY + "px";
      }
      if (humanTextRef.current) {
        humanTextRef.current.style.clipPath = `circle(${revealSize}px at ${cursorX}px ${cursorY}px)`;
      }

      animationId = requestAnimationFrame(animate);
    };

    const handleHoverEnter = () => {
      if (cursorRingRef.current) {
        cursorRingRef.current.style.width = "40px";
        cursorRingRef.current.style.height = "40px";
        cursorRingRef.current.style.borderColor = "#ff00ff";
      }
      targetRevealSize = 80;
    };

    const handleHoverLeave = () => {
      if (cursorRingRef.current) {
        cursorRingRef.current.style.width = "24px";
        cursorRingRef.current.style.height = "24px";
        cursorRingRef.current.style.borderColor = "#00ffff";
      }
      targetRevealSize = 60;
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseleave", handleMouseLeave);

    // Add hover listeners
    setTimeout(() => {
      const interactiveElements = document.querySelectorAll(
        'button, input, a, [class*="featureCard"]',
      );
      interactiveElements.forEach((el) => {
        el.addEventListener("mouseenter", handleHoverEnter);
        el.addEventListener("mouseleave", handleHoverLeave);
      });
    }, 100);

    animate();

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseleave", handleMouseLeave);
      if (animationId) cancelAnimationFrame(animationId);
    };
  }, []);

  // Create floating elements
  useEffect(() => {
    const floatingContainer = document.getElementById("floatingElements");
    const particlesContainer = document.getElementById("particles");

    if (floatingContainer && floatingContainer.children.length === 0) {
      const shapeTypes = ["hex", "ring", "diamond"];
      for (let i = 0; i < 8; i++) {
        const shape = document.createElement("div");
        shape.className = `${styles.floatShape} ${styles[shapeTypes[i % 3]]}`;
        shape.style.left = Math.random() * 90 + 5 + "%";
        shape.style.top = Math.random() * 90 + 5 + "%";
        floatingContainer.appendChild(shape);
      }
    }

    if (particlesContainer && particlesContainer.children.length === 0) {
      const colors = ["#00ffff", "#ff00ff", "#8b00ff", "#ff0066"];
      for (let i = 0; i < 25; i++) {
        const p = document.createElement("div");
        p.className = styles.particle;
        const size = Math.random() * 3 + 2;
        const color = colors[Math.floor(Math.random() * colors.length)];
        p.style.width = size + "px";
        p.style.height = size + "px";
        p.style.left = Math.random() * 100 + "%";
        p.style.background = color;
        p.style.boxShadow = `0 0 ${size * 2}px ${color}`;
        p.style.animationDelay = Math.random() * 20 + "s";
        p.style.animationDuration = 15 + Math.random() * 10 + "s";
        particlesContainer.appendChild(p);
      }
    }
  }, []);

  const handleCtaClick = (e) => {
    e.preventDefault();

    goToSection(3);
  };

  // Get section visibility class
  const getSectionClass = (index) => {
    if (index === currentSection) return styles.sectionActive;
    if (index < currentSection) return styles.sectionAbove;
    return styles.sectionBelow;
  };

  return (
    <>
      <Head>
        <title>MANAVAI - AI Detector & Humanizer</title>
        <meta
          name="description"
          content="Transform synthetic text into authentic human expression"
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      {/* Custom Cursor */}
      <div className={styles.cursorGlow} ref={cursorGlowRef}></div>
      <div className={styles.cursorRing} ref={cursorRingRef}></div>
      <div className={styles.cursorDot} ref={cursorDotRef}></div>

      {/* Progress Indicator */}
      <div className={styles.progressBar}>
        <div
          className={styles.progressFill}
          style={{
            height: `${((currentSection + 1) / TOTAL_SECTIONS) * 100}%`,
          }}
        ></div>
      </div>

      {/* Section Dots */}
      <div className={styles.sectionDots}>
        {[0, 1, 2, 3].map((i) => (
          <button
            key={i}
            className={`${styles.dot} ${currentSection === i ? styles.dotActive : ""}`}
            onClick={() => goToSection(i)}
            aria-label={`Go to section ${i + 1}`}
          />
        ))}
      </div>

      {/* Background Layers */}
      <div className={styles.gradientBg} ref={gradientBgRef}></div>
      <div className={styles.warmBgReveal} ref={warmBgRevealRef}></div>
      <div className={styles.cyberGrid}></div>

      {/* Text Layers */}
      <div className={styles.textRevealContainer}>
        <div className={styles.aiTextLayer}>{aiText}</div>
        <div className={styles.humanTextLayer} ref={humanTextRef}>
          {humanText}
        </div>
      </div>

      {/* Decorative Elements */}
      <div className={`${styles.neonLine} ${styles.neonLine1}`}></div>
      <div className={`${styles.neonLine} ${styles.neonLine2}`}></div>
      <div className={styles.dataStream} style={{ left: "5%" }}></div>
      <div
        className={styles.dataStream}
        style={{ left: "95%", animationDelay: "2s" }}
      ></div>
      <div id="floatingElements" className={styles.floatingElements}></div>
      <div id="particles" className={styles.particles}></div>
      <div className={styles.scanlines}></div>

      {/* Main Container */}
      <div className={styles.fullpageContainer} ref={containerRef}>
        {/* Section 0: Hero */}
        <section className={`${styles.section} ${getSectionClass(0)}`}>
          <div className={styles.heroContent}>
            <h1 className={styles.logo} ref={logoRef}>
              MANAVAI
            </h1>
            <p className={styles.tagline}>AI Detector & Humanizer</p>
          </div>
          <div className={styles.scrollHint}>
            <div className={styles.scrollArrow}></div>
            <span>SCROLL TO EXPLORE</span>
          </div>
        </section>

        {/* Section 1: Features */}
        <section className={`${styles.section} ${getSectionClass(1)}`}>
          <h2 className={styles.sectionTitle}>Why Choose MANAVAI?</h2>
          <div className={styles.featuresGrid}>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>🎯</div>
              <h3>??% ACCURACY</h3>
              <p>
                Advanced neural networks detect AI-generated content with
                precision.
              </p>
            </div>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>⚡</div>
              <h3>INSTANT RESULTS</h3>
              <p>Get detection and humanization in milliseconds.</p>
            </div>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>🔒</div>
              <h3>SECURE & PRIVATE</h3>
              <p>Your content is encrypted and never stored.</p>
            </div>
            <div className={styles.featureCard}>
              <div className={styles.featureIcon}>✨</div>
              <h3>PRESERVE STYLE </h3>
              <p>Humanize without losing your unique style.</p>
            </div>
          </div>
        </section>

        {/* Section 2: CTA */}
        <section className={`${styles.section} ${getSectionClass(2)}`}>
          <div className={styles.ctaContent}>
            <h2 className={styles.ctaTitle}>Ready to Transform?</h2>
            <p className={styles.ctaDescription}>
              Join over ???? users who trust MANAVAI to bridge artificial and
              authentic.
            </p>
            <Link href="/login">
              <button className={styles.ctaBtn}>GET STARTED</button>
            </Link>
          </div>
        </section>

        {/* Section 3: Login */}
        <section className={`${styles.section} ${getSectionClass(3)}`}>
          <div className={styles.loginContainer}>
            <div className={styles.loginInfo}>
              <h2>
                Enter The
                <br />
                Digital Realm
              </h2>
              <p>Create your account and start transforming content today.</p>
              <ul className={styles.featureList}>
                <li>Free trial with 10,000 words</li>
                <li>No credit card required</li>
                <li>Cancel anytime</li>
                <li>24/7 Support</li>
              </ul>
            </div>

            <div className={styles.loginFormContainer}>
              <div className={styles.formTabs}>
                <button
                  className={`${styles.formTab} ${activeTab === "login" ? styles.active : ""}`}
                  onClick={() => setActiveTab("login")}
                >
                  LOGIN
                </button>
                <button
                  className={`${styles.formTab} ${activeTab === "signup" ? styles.active : ""}`}
                  onClick={() => setActiveTab("signup")}
                >
                  SIGN UP
                </button>
              </div>

              {activeTab === "login" && (
                <form className={styles.formPanel}>
                  <div className={styles.formGroup}>
                    <label className={styles.formLabel}>EMAIL</label>
                    <input
                      type="email"
                      className={styles.formInput}
                      placeholder="Enter your email"
                    />
                  </div>
                  <div className={styles.formGroup}>
                    <label className={styles.formLabel}>PASSWORD</label>
                    <input
                      type="password"
                      className={styles.formInput}
                      placeholder="Enter your password"
                    />
                  </div>
                  <button type="submit" className={styles.formSubmit}>
                    JACK IN
                  </button>
                  <div className={styles.formDivider}>
                    <span>OR</span>
                  </div>
                  <div className={styles.socialLogin}>
                    <button type="button" className={styles.socialBtn}>
                      Google
                    </button>
                    <button type="button" className={styles.socialBtn}>
                      GitHub
                    </button>
                  </div>
                </form>
              )}

              {activeTab === "signup" && (
                <form className={styles.formPanel}>
                  <div className={styles.formGroup}>
                    <label className={styles.formLabel}>USERNAME</label>
                    <input
                      type="text"
                      className={styles.formInput}
                      placeholder="Choose your handle"
                    />
                  </div>
                  <div className={styles.formGroup}>
                    <label className={styles.formLabel}>EMAIL</label>
                    <input
                      type="email"
                      className={styles.formInput}
                      placeholder="Enter your email"
                    />
                  </div>
                  <div className={styles.formGroup}>
                    <label className={styles.formLabel}>PASSWORD</label>
                    <input
                      type="password"
                      className={styles.formInput}
                      placeholder="Create password"
                    />
                  </div>
                  <button type="submit" className={styles.formSubmit}>
                    INITIALIZE
                  </button>
                  <div className={styles.formDivider}>
                    <span>OR</span>
                  </div>
                  <div className={styles.socialLogin}>
                    <button type="button" className={styles.socialBtn}>
                      Google
                    </button>
                    <button type="button" className={styles.socialBtn}>
                      GitHub
                    </button>
                  </div>
                </form>
              )}
            </div>
          </div>
        </section>
      </div>
    </>
  );
}
