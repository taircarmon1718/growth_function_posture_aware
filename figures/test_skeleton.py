#!/usr/bin/env python3
"""Quick test of skeleton_length_figure.py"""

import sys
from pathlib import Path

# Add figures directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("TESTING SKELETON LENGTH FIGURE")
print("=" * 60)

try:
    # Import main
    from skeleton_length_figure import main
    
    print("\n✓ Imports successful")
    print("\nRunning main()...\n")
    
    # Run main
    main()
    
    print("\n✓ Script completed successfully")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

