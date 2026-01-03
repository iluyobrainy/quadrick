#!/usr/bin/env python3
"""
Cache Cleaner Script
Run this anytime to clear all __pycache__ directories
"""

import os
import shutil

def clear_pycache():
    """Clear all __pycache__ directories in the project"""
    count = 0
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(pycache_path)
                print(f'🗑️  Removed: {pycache_path}')
                count += 1
            except Exception as e:
                print(f'⚠️  Failed to remove: {pycache_path} - {e}')

    if count == 0:
        print('✅ No __pycache__ directories found')
    else:
        print(f'✅ Cleared {count} __pycache__ directories')

if __name__ == "__main__":
    print("🧹 CLEARING PYTHON CACHE...")
    print("=" * 40)
    clear_pycache()
    print("\\n🎯 Run 'python clear_cache.py' anytime to clear cache!")
