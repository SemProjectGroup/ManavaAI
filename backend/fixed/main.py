#!/usr/bin/env python3
"""
ManavAI - AI Content Detection & Humanization Tool
Version 3.0 - Ultimate
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Ensure we can import from local modules
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║   ███╗   ███╗ █████╗ ███╗   ██╗ █████╗ ██╗   ██╗ █████╗ ██╗  ║
    ║   ████╗ ████║██╔══██╗████╗  ██║██╔══██╗██║   ██║██╔══██╗██║  ║
    ║   ██╔████╔██║███████║██╔██╗ ██║███████║██║   ██║███████║██║  ║
    ║   ██║╚██╔╝██║██╔══██║██║╚██╗██║██╔══██║╚██╗ ██╔╝██╔══██║██║  ║
    ║   ██║ ╚═╝ ██║██║  ██║██║ ╚████║██║  ██║ ╚████╔╝ ██║  ██║██║  ║
    ║   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝╚═╝  ║
    ║                                                              ║
    ║          AI Content Detection & Humanization Tool            ║
    ║                    Version 3.0 - Ultimate                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def get_multiline_input():
    """Helper to get multiline input from user."""
    print("\n📝 Enter/paste text (Press Enter twice when done):\n")
    lines = []
    empty_count = 0
    while empty_count < 2:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            empty_count += 1
        else:
            empty_count = 0
            lines.append(line)
    return "\n".join(lines).strip()


def detect_command(args):
    """Handle the detect command."""
    try:
        from detector_v3 import ModernAIDetector
    except ImportError:
        print("❌ Error: detector_v3.py not found.")
        return 1
    
    print("\n🔍 Initializing AI Detector...")
    detector = ModernAIDetector()
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return 1
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📄 Loaded from: {args.file}")
    elif args.text:
        text = args.text
    else:
        print("❌ Provide text with --text or --file")
        return 1
    
    if len(text.split()) < 5:
        print("❌ Text too short.")
        return 1

    print(f"📝 Analyzing {len(text.split())} words...")
    
    if args.detailed:
        print(detector.get_detailed_report(text))
    else:
        result = detector.detect(text, detailed=True)
        
        print("\n" + "=" * 55)
        print("📊 DETECTION RESULT")
        print("=" * 55)
        print(f"   Verdict:           {result['verdict'].replace('_', ' ').upper()}")
        print(f"   AI Probability:    {result['ai_percentage']}%")
        print(f"   Human Probability: {result['human_percentage']}%")
        print(f"   Confidence:        {result['confidence'].title()}")
        
        if result.get('detection_reasons'):
            print(f"\n   Key factors:")
            for reason in result['detection_reasons'][:4]:
                print(f"      • {reason}")
        
        print("=" * 55)
    
    return 0


def humanize_command(args):
    """Handle the humanize command."""
    try:
        from inference.humanizer_inference import TextHumanizer
    except ImportError:
        print("❌ Error: inference/humanizer_inference.py not found.")
        print("Make sure you have run the training scripts first.")
        return 1

    print("\n🧠 Initializing Text Humanizer...")
    humanizer = TextHumanizer()

    if args.file:
        if not os.path.exists(args.file):
            print(f"❌ File not found: {args.file}")
            return 1
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📄 Loaded from: {args.file}")
    elif args.text:
        text = args.text
    else:
        print("❌ Provide text with --text or --file")
        return 1

    print(f"✍️  Humanizing ({args.style} style, {args.intensity} intensity)...")
    
    # Process
    result = humanizer.humanize(text, style=args.style, intensity=args.intensity)
    
    # Output
    print("\n" + "=" * 55)
    print("✨ HUMANIZED TEXT")
    print("=" * 55)
    print(result)
    print("=" * 55)

    # Save if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n💾 Saved output to: {args.output}")

    return 0


def interactive_mode():
    """Interactive mode."""
    print_banner()
    
    # Models are loaded lazily
    detector = None
    humanizer = None

    while True:
        print("\n" + "=" * 50)
        print("1. Detect AI Content")
        print("2. Detect with Detailed Report")
        # print("3. Humanize Text (Rewrite)")
        # print("4. Auto (Detect -> Humanize -> Verify)")
        print("5. Exit")
        print("=" * 50)
        
        choice = input("\nChoice: ").strip()
        
        if choice == '5':
            print("\n👋 Goodbye!")
            break

        # Text Input Logic
        if choice in ['1', '2', '3', '4']:
            text = get_multiline_input()
            
            if len(text.split()) < 5:
                print("❌ Text too short. Need at least 5 words.")
                continue

        # --- OPTION 1 & 2: DETECTION ---
        if choice in ['1', '2']:
            if detector is None:
                print("\n🔍 Loading detector...")
                from detector_v3 import ModernAIDetector
                detector = ModernAIDetector()
            
            print("\n⏳ Analyzing...")
            if choice == '2':
                print(detector.get_detailed_report(text))
            else:
                result = detector.detect(text)
                print("\n" + "=" * 50)
                print(f"🎯 Verdict:     {result['verdict'].replace('_', ' ').upper()}")
                print(f"📊 AI Score:    {result['ai_percentage']}%")
                print(f"📊 Human Score: {result['human_percentage']}%")
                print("=" * 50)

        # --- OPTION 3: HUMANIZATION ---
        elif choice == '3':
            if humanizer is None:
                print("\n🧠 Loading humanizer...")
                try:
                    from inference.humanizer_inference import TextHumanizer
                    humanizer = TextHumanizer()
                except ImportError:
                    print("❌ Error: humanizer files not found.")
                    continue

            style = input("   Style (casual/formal/academic) [casual]: ").strip() or 'casual'
            intensity = input("   Intensity (light/medium/aggressive) [medium]: ").strip() or 'medium'
            
            print("\n✍️  Humanizing...")
            output = humanizer.humanize(text, style=style, intensity=intensity)
            
            print("\n" + "=" * 55)
            print("✨ RESULT")
            print("=" * 55)
            print(output)
            print("=" * 55)

        # --- OPTION 4: AUTO PIPELINE ---
         # --- OPTION 4: AUTO PIPELINE ---
        elif choice == '4':
            if detector is None:
                print("\n🔍 Loading detector...")
                from detector_v3 import ModernAIDetector
                detector = ModernAIDetector()
            if humanizer is None:
                print("\n🧠 Loading humanizer...")
                try:
                    from inference.humanizer_inference import TextHumanizer
                    humanizer = TextHumanizer()
                except ImportError:
                    print("❌ Error: humanizer files not found.")
                    continue

            # 1. Initial Check
            print("\n1️⃣  Initial Detection...")
            initial_res = detector.detect(text, detailed=True)
            print(f"    AI Score: {initial_res['ai_percentage']}%")
            print(f"    AI Phrases Found: {initial_res.get('analysis', {}).get('ai_phrase_count', 0)}")

            if initial_res['ai_percentage'] < 40:
                print("    ✅ Text already appears human. Skipping.")
            else:
                # 2. Humanize - ALWAYS use aggressive for high scores
                print("\n2️⃣  Humanizing text (aggressive mode)...")
                new_text = humanizer.humanize(text, style='casual', intensity='aggressive')
                
                # 3. If still high, humanize again
                mid_res = detector.detect(new_text)
                if mid_res['ai_percentage'] > 60:
                    print("    Still high, applying second pass...")
                    new_text = humanizer.humanize(new_text, style='casual', intensity='aggressive')

                # 4. Final Check
                print("\n3️⃣  Verifying result...")
                final_res = detector.detect(new_text, detailed=True)
                
                reduction = initial_res['ai_percentage'] - final_res['ai_percentage']
                
                print("\n" + "=" * 60)
                print("🚀 OPTIMIZATION COMPLETE")
                print("=" * 60)
                print(f"📉 Score: {initial_res['ai_percentage']}% -> {final_res['ai_percentage']}% (↓{reduction:.1f}%)")
                print(f"📝 AI Phrases: {initial_res.get('analysis', {}).get('ai_phrase_count', 0)} -> {final_res.get('analysis', {}).get('ai_phrase_count', 0)}")
                print("-" * 60)
                print(new_text)
                print("=" * 60)
                
                if final_res['ai_percentage'] > 50:
                    print("\n⚠️  Score still above 50%. Tips:")
                    print("    - Try running again for another pass")
                    print("    - Manually edit remaining formal phrases")
                    if final_res.get('ai_phrases_found'):
                        print(f"    - Remaining AI phrases: {final_res['ai_phrases_found'][:3]}")


def main():
    parser = argparse.ArgumentParser(description='ManavAI - AI Content Detection & Humanization')
    subparsers = parser.add_subparsers(dest='command')
    
    # Detect Command
    detect_p = subparsers.add_parser('detect', help='Detect AI content')
    detect_p.add_argument('--text', '-t', type=str, help='Text to analyze')
    detect_p.add_argument('--file', '-f', type=str, help='File to analyze')
    detect_p.add_argument('--detailed', '-d', action='store_true', help='Detailed report')
    
    # Humanize Command
    human_p = subparsers.add_parser('humanize', help='Humanize AI text')
    human_p.add_argument('--text', '-t', type=str, help='Text to humanize')
    human_p.add_argument('--file', '-f', type=str, help='File to humanize')
    human_p.add_argument('--style', '-s', type=str, default='casual', choices=['casual', 'formal', 'academic'], help='Writing style')
    human_p.add_argument('--intensity', '-i', type=str, default='medium', choices=['light', 'medium', 'aggressive'], help='Transformation intensity')
    human_p.add_argument('--output', '-o', type=str, help='Save output to file')

    # Interactive Command
    subparsers.add_parser('interactive', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.command == 'detect':
        return detect_command(args)
    elif args.command == 'humanize':
        return humanize_command(args)
    else:
        interactive_mode()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())