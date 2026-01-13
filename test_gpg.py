import os
import shutil
import subprocess
import tempfile
import pytest

@pytest.fixture
def gpg_home():
    """Create a temporary GPG home directory."""
    temp_dir = tempfile.mkdtemp()
    os.environ["GNUPGHOME"] = temp_dir
    os.chmod(temp_dir, 0o700)
    yield temp_dir
    shutil.rmtree(temp_dir)
    if "GNUPGHOME" in os.environ:
        del os.environ["GNUPGHOME"]

def generate_key(name, email, passphrase=None):
    """Generate a GPG key non-interactively."""
    # GPG batch generation configuration
    batch_config = f"""
    Key-Type: RSA
    Key-Length: 2048
    Subkey-Type: RSA
    Subkey-Length: 2048
    Name-Real: {name}
    Name-Email: {email}
    Expire-Date: 0
    %no-protection
    %commit
    """
    
    # If passphrase provided (though %no-protection avoids it for tests usually)
    # Ideally we use no-protection for automated tests unless testing passphrase logic explicitly
    
    cmd = ["gpg", "--batch", "--generate-key"]
    
    proc = subprocess.run(
        cmd, 
        input=batch_config, 
        text=True, 
        capture_output=True
    )
    
    if proc.returncode != 0:
        raise RuntimeError(f"Key generation failed: {proc.stderr}")
    
    return proc.stdout

def test_gpg_key_generation_and_operations(gpg_home):
    """
    Test generating two keys and performing operations.
    Key 1: Full Name: Gemini 3 Pro
    Key 2: Full Name: Test User 2
    """
    
    # 1. Generate Key 1
    print("Generating Key 1...")
    generate_key("Gemini 3 Pro", "gemini3pro@example.com")
    
    # 2. Generate Key 2
    print("Generating Key 2...")
    generate_key("Test User 2", "testuser2@example.com")
    
    # 3. List keys to verify
    list_cmd = subprocess.run(
        ["gpg", "--list-keys"], 
        capture_output=True, 
        text=True
    )
    assert list_cmd.returncode == 0
    assert "Gemini 3 Pro" in list_cmd.stdout
    assert "Test User 2" in list_cmd.stdout
    
    # 4. Create a test file
    test_file = os.path.join(gpg_home, "test_data.txt")
    with open(test_file, "w") as f:
        f.write("Hello GPG World")
        
    # 5. Detach sign with Key 1 (Gemini 3 Pro)
    # We need to find the Key ID or use the name
    # Using name as key_id
    sign_cmd = subprocess.run(
        ["gpg", "--batch", "--yes", "--local-user", "Gemini 3 Pro", "--armor", "--detach-sign", test_file],
        capture_output=True,
        text=True
    )
    assert sign_cmd.returncode == 0
    assert os.path.exists(test_file + ".asc")
    
    # 6. Export Public Key 1 in strict OpenPGP mode
    export_cmd = subprocess.run(
        ["gpg", "--batch", "--yes", "--openpgp", "--armor", "--export", "Gemini 3 Pro"],
        capture_output=True,
        text=True
    )
    assert export_cmd.returncode == 0
    assert "-----BEGIN PGP PUBLIC KEY BLOCK-----" in export_cmd.stdout
    
    print("Test completed successfully.")

if __name__ == "__main__":
    # Allow running directly for quick check
    # Manually setup context if running as script
    temp_dir = tempfile.mkdtemp()
    try:
        os.environ["GNUPGHOME"] = temp_dir
        os.chmod(temp_dir, 0o700)
        test_gpg_key_generation_and_operations(temp_dir)
    finally:
        shutil.rmtree(temp_dir)

