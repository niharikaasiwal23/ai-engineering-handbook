#!/bin/bash
# Build script for the AI Engineering Handbook (mdBook)
# 
# Usage:
#   ./build.sh setup    — Copy chapters into src/ and install dependencies
#   ./build.sh serve    — Start local development server (hot reload)
#   ./build.sh build    — Build the static site into output/
#   ./build.sh clean    — Remove generated files

set -e

BOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$BOOK_DIR/src"

copy_chapters() {
    echo "📖 Copying chapters into src/..."
    
    cp "$BOOK_DIR/chapter-01-ai-landscape/chapter-01.md" "$SRC_DIR/chapter-01.md"
    cp "$BOOK_DIR/chapter-02-llm-fundamentals/chapter-02.md" "$SRC_DIR/chapter-02.md"
    cp "$BOOK_DIR/chapter-03-apis/chapter-03.md" "$SRC_DIR/chapter-03.md"
    cp "$BOOK_DIR/chapter-04-prompt-engineering/chapter-04.md" "$SRC_DIR/chapter-04.md"
    cp "$BOOK_DIR/chapter-05-building-ai-apps/chapter-05.md" "$SRC_DIR/chapter-05.md"
    cp "$BOOK_DIR/chapter-06-embeddings/chapter-06.md" "$SRC_DIR/chapter-06.md"
    cp "$BOOK_DIR/chapter-07-rag/chapter-07.md" "$SRC_DIR/chapter-07.md"
    cp "$BOOK_DIR/chapter-08-databases/chapter-08.md" "$SRC_DIR/chapter-08.md"
    cp "$BOOK_DIR/chapter-09-ai-automation/chapter-09.md" "$SRC_DIR/chapter-09.md"
    cp "$BOOK_DIR/chapter-10-ai-agents/chapter-10.md" "$SRC_DIR/chapter-10.md"
    cp "$BOOK_DIR/chapter-11-flutter-ai/chapter-11.md" "$SRC_DIR/chapter-11.md"
    cp "$BOOK_DIR/chapter-12-deployment/chapter-12.md" "$SRC_DIR/chapter-12.md"
    cp "$BOOK_DIR/chapter-13-client-projects/chapter-13.md" "$SRC_DIR/chapter-13.md"
    cp "$BOOK_DIR/chapter-14-freelancing/chapter-14.md" "$SRC_DIR/chapter-14.md"
    cp "$BOOK_DIR/chapter-15-ai-saas/chapter-15.md" "$SRC_DIR/chapter-15.md"
    cp "$BOOK_DIR/chapter-16-nodejs-ai/chapter-16.md" "$SRC_DIR/chapter-16.md"
    cp "$BOOK_DIR/chapter-17-aws-ai/chapter-17.md" "$SRC_DIR/chapter-17.md"
    cp "$BOOK_DIR/BOOK_RULES.md" "$SRC_DIR/book-rules.md"
    
    echo "✅ Chapters copied."
}

case "${1:-}" in
    setup)
        echo "🔧 Setting up mdBook..."
        
        # Check if mdbook is installed
        if ! command -v mdbook &> /dev/null; then
            echo "Installing mdbook..."
            brew install mdbook
        fi
        
        # Check if mdbook-mermaid is installed
        if ! command -v mdbook-mermaid &> /dev/null; then
            echo "Installing mdbook-mermaid..."
            echo "Run: cargo install mdbook-mermaid"
            echo "  OR download from: https://github.com/badboy/mdbook-mermaid/releases"
            echo ""
            echo "If you don't have cargo, install Rust first: brew install rust"
        fi
        
        # Create theme directory
        mkdir -p "$BOOK_DIR/theme"
        
        # Copy chapters
        copy_chapters
        
        # Initialize mermaid support
        if command -v mdbook-mermaid &> /dev/null; then
            cd "$BOOK_DIR" && mdbook-mermaid install .
        fi
        
        echo ""
        echo "✅ Setup complete! Run: ./build.sh serve"
        ;;
    
    serve)
        copy_chapters
        echo "🚀 Starting development server..."
        echo "   Open: http://localhost:3000"
        cd "$BOOK_DIR" && mdbook serve --port 3000 --open
        ;;
    
    build)
        copy_chapters
        echo "📦 Building static site..."
        cd "$BOOK_DIR" && mdbook build
        echo "✅ Built to: $BOOK_DIR/output/"
        echo "   Open: output/index.html"
        ;;
    
    clean)
        echo "🧹 Cleaning generated files..."
        rm -rf "$BOOK_DIR/output"
        rm -f "$SRC_DIR"/chapter-*.md
        rm -f "$SRC_DIR/book-rules.md"
        echo "✅ Cleaned."
        ;;
    
    *)
        echo "AI Engineering Handbook — Build Script"
        echo ""
        echo "Usage:"
        echo "  ./build.sh setup    Install dependencies and prepare source files"
        echo "  ./build.sh serve    Start local dev server with hot reload"
        echo "  ./build.sh build    Build static site to output/"
        echo "  ./build.sh clean    Remove generated files"
        ;;
esac
