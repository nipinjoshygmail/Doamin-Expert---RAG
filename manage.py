#!/usr/bin/env python3
"""
RAG Flask Application - Management CLI
Utility script for common management tasks
"""

import os
import sys
import shutil
import argparse
from datetime import datetime


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def check_env():
    """Check if environment is properly configured"""
    print_header("Environment Check")
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("❌ Python 3.8+ required")
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment active")
    else:
        issues.append("⚠️  Virtual environment not activated")
    
    # Check .env file
    if os.path.exists('.env'):
        print("✅ .env file exists")
        
        # Check for API key
        with open('.env', 'r') as f:
            content = f.read()
            if 'GROQ_API_KEY' in content and 'your_groq' not in content.lower():
                print("✅ GROQ_API_KEY configured")
            else:
                issues.append("⚠️  GROQ_API_KEY not configured in .env")
    else:
        issues.append("❌ .env file missing")
    
    # Check required directories
    dirs = ['uploads', 'chroma_db', 'templates', 'static', 'utils']
    for d in dirs:
        if os.path.exists(d):
            print(f"✅ {d}/ directory exists")
        else:
            issues.append(f"⚠️  {d}/ directory missing")
    
    # Check required files
    files = ['app.py', 'requirements.txt', 'utils/transcription.py', 'utils/loaders.py']
    for f in files:
        if os.path.exists(f):
            print(f"✅ {f} exists")
        else:
            issues.append(f"❌ {f} missing")
    
    # Check FFmpeg
    if shutil.which('ffmpeg'):
        print("✅ FFmpeg installed")
    else:
        issues.append("⚠️  FFmpeg not found (required for audio)")
    
    # Summary
    print("\n" + "-" * 60)
    if issues:
        print(f"\n⚠️  Found {len(issues)} issue(s):\n")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ All checks passed! Environment is ready.")
        return True


def reset_database():
    """Reset the vector database"""
    print_header("Reset Database")
    
    if os.path.exists('chroma_db'):
        confirm = input("⚠️  This will delete all indexed documents. Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            shutil.rmtree('chroma_db')
            print("✅ Database deleted")
            os.makedirs('chroma_db', exist_ok=True)
            print("✅ Fresh database directory created")
        else:
            print("❌ Operation cancelled")
    else:
        print("ℹ️  No database found")


def clear_uploads():
    """Clear temporary upload directory"""
    print_header("Clear Uploads")
    
    if os.path.exists('uploads'):
        files = os.listdir('uploads')
        if files:
            confirm = input(f"⚠️  Delete {len(files)} file(s)? (yes/no): ")
            if confirm.lower() == 'yes':
                shutil.rmtree('uploads')
                os.makedirs('uploads', exist_ok=True)
                print(f"✅ Deleted {len(files)} file(s)")
            else:
                print("❌ Operation cancelled")
        else:
            print("ℹ️  Upload directory is empty")
    else:
        os.makedirs('uploads', exist_ok=True)
        print("✅ Upload directory created")


def backup_database():
    """Backup the vector database"""
    print_header("Backup Database")
    
    if not os.path.exists('chroma_db'):
        print("❌ No database found to backup")
        return
    
    # Create backups directory
    os.makedirs('backups', exist_ok=True)
    
    # Create backup with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'chroma_backup_{timestamp}'
    backup_path = os.path.join('backups', backup_name)
    
    print(f"📦 Creating backup: {backup_name}")
    shutil.copytree('chroma_db', backup_path)
    
    # Compress backup
    shutil.make_archive(backup_path, 'gztar', 'backups', backup_name)
    shutil.rmtree(backup_path)
    
    print(f"✅ Backup created: {backup_name}.tar.gz")
    
    # Show all backups
    backups = [f for f in os.listdir('backups') if f.endswith('.tar.gz')]
    print(f"\n📚 Total backups: {len(backups)}")
    for backup in sorted(backups, reverse=True)[:5]:
        size = os.path.getsize(os.path.join('backups', backup))
        print(f"  - {backup} ({size // 1024}KB)")


def show_stats():
    """Show application statistics"""
    print_header("Application Statistics")
    
    # Database stats
    if os.path.exists('chroma_db'):
        db_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk('chroma_db')
            for filename in filenames
        )
        print(f"📊 Database size: {db_size / 1024 / 1024:.2f} MB")
    else:
        print("📊 Database: Not initialized")
    
    # Upload stats
    if os.path.exists('uploads'):
        uploads = os.listdir('uploads')
        print(f"📁 Temporary files: {len(uploads)}")
    
    # Backup stats
    if os.path.exists('backups'):
        backups = [f for f in os.listdir('backups') if f.endswith('.tar.gz')]
        total_size = sum(
            os.path.getsize(os.path.join('backups', f))
            for f in backups
        )
        print(f"💾 Backups: {len(backups)} ({total_size / 1024 / 1024:.2f} MB)")
    
    # Log stats
    if os.path.exists('logs'):
        logs = os.listdir('logs')
        print(f"📝 Log files: {len(logs)}")


def clean_all():
    """Clean all temporary data"""
    print_header("Clean All Temporary Data")
    
    confirm = input("⚠️  This will clear uploads and logs. Continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("❌ Operation cancelled")
        return
    
    # Clear uploads
    if os.path.exists('uploads'):
        shutil.rmtree('uploads')
        os.makedirs('uploads', exist_ok=True)
        print("✅ Uploads cleared")
    
    # Clear logs
    if os.path.exists('logs'):
        shutil.rmtree('logs')
        os.makedirs('logs', exist_ok=True)
        print("✅ Logs cleared")
    
    # Clear Python cache
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            shutil.rmtree(os.path.join(root, '__pycache__'))
    print("✅ Python cache cleared")
    
    print("\n✅ Cleanup complete!")


def create_systemd_service():
    """Generate systemd service file"""
    print_header("Generate Systemd Service")
    
    username = input("Enter username (default: current user): ").strip()
    if not username:
        username = os.getenv('USER', 'ubuntu')
    
    workdir = os.getcwd()
    venv_python = os.path.join(workdir, 'venv', 'bin', 'gunicorn')
    
    service_content = f"""[Unit]
Description=RAG Flask Application
After=network.target

[Service]
User={username}
WorkingDirectory={workdir}
Environment="PATH={os.path.join(workdir, 'venv', 'bin')}"
ExecStart={venv_python} -w 4 -b 127.0.0.1:5000 --timeout 120 wsgi:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    filename = 'rag-app.service'
    with open(filename, 'w') as f:
        f.write(service_content)
    
    print(f"✅ Service file created: {filename}")
    print("\nTo install:")
    print(f"  sudo cp {filename} /etc/systemd/system/")
    print("  sudo systemctl daemon-reload")
    print("  sudo systemctl enable rag-app")
    print("  sudo systemctl start rag-app")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='RAG Flask Application Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage.py check        Check environment configuration
  python manage.py reset        Reset vector database
  python manage.py backup       Backup database
  python manage.py stats        Show statistics
  python manage.py clean        Clean temporary data
  python manage.py systemd      Generate systemd service file
        """
    )
    
    parser.add_argument(
        'command',
        choices=['check', 'reset', 'clear', 'backup', 'stats', 'clean', 'systemd'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'check':
        check_env()
    elif args.command == 'reset':
        reset_database()
    elif args.command == 'clear':
        clear_uploads()
    elif args.command == 'backup':
        backup_database()
    elif args.command == 'stats':
        show_stats()
    elif args.command == 'clean':
        clean_all()
    elif args.command == 'systemd':
        create_systemd_service()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)