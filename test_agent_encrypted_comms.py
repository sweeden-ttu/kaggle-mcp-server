import os
import shutil
import subprocess
import tempfile
import pytest

@pytest.fixture
def gpg_context():
    """Create a temporary GPG home directory."""
    temp_dir = tempfile.mkdtemp()
    os.environ["GNUPGHOME"] = temp_dir
    os.chmod(temp_dir, 0o700)
    
    # Create plain_message.txt as requested
    with open("plain_message.txt", "w") as f:
        f.write("This is a plain text message.")
        
    yield temp_dir
    
    shutil.rmtree(temp_dir)
    if "GNUPGHOME" in os.environ:
        del os.environ["GNUPGHOME"]
    
    # Cleanup files
    for f in ["plain_message.txt", "message_for_agent2.txt", "message_for_agent2.txt.asc", "decrypted_message.txt"]:
        if os.path.exists(f):
            os.remove(f)

def generate_key_with_passphrase(name, email, passphrase):
    """Generate a GPG key with a specific passphrase."""
    batch_config = f"""
    Key-Type: RSA
    Key-Length: 2048
    Subkey-Type: RSA
    Subkey-Length: 2048
    Name-Real: {name}
    Name-Email: {email}
    Expire-Date: 0
    Passphrase: {passphrase}
    %commit
    """
    
    cmd = ["gpg", "--batch", "--pinentry-mode", "loopback", "--generate-key"]
    
    proc = subprocess.run(
        cmd, 
        input=batch_config, 
        text=True, 
        capture_output=True
    )
    
    if proc.returncode != 0:
        raise RuntimeError(f"Key generation failed for {name}: {proc.stderr}")
    
    return proc.stdout

def test_encrypted_communication(gpg_context):
    """
    End-to-end test for encrypted communication between Agent1 and Agent2.
    """
    pass1 = "agent1_secret_passphrase"
    pass2 = "agent2_secret_passphrase"
    
    print("1. Generating Agent1 Key...")
    generate_key_with_passphrase("Agent1", "agent1@example.com", pass1)
    
    print("2. Generating Agent2 Key...")
    generate_key_with_passphrase("Agent2", "agent2@example.com", pass2)
    
    # Verify keys exist
    list_cmd = subprocess.run(["gpg", "--list-keys"], capture_output=True, text=True)
    assert "Agent1" in list_cmd.stdout
    assert "Agent2" in list_cmd.stdout
    
    print("3. Creating message_for_agent2.txt...")
    message_content = "Confidential message for Agent 2"
    with open("message_for_agent2.txt", "w") as f:
        f.write(message_content)
        
    print("4. Encrypting message for Agent2 (ASCII armored)...")
    # --trust-model always avoids interactive trust validation for test keys
    encrypt_cmd = subprocess.run(
        [
            "gpg", "--batch", "--yes", 
            "--trust-model", "always",
            "--recipient", "Agent2",
            "--armor", 
            "--encrypt", 
            "message_for_agent2.txt"
        ],
        capture_output=True,
        text=True
    )
    
    if encrypt_cmd.returncode != 0:
        print(f"Encryption failed: {encrypt_cmd.stderr}")
    assert encrypt_cmd.returncode == 0
    assert os.path.exists("message_for_agent2.txt.asc")
    
    print("5. Sending and Decrypting as Agent2...")
    # Decrypt using Agent2's passphrase
    decrypt_cmd = subprocess.run(
        [
            "gpg", "--batch", "--yes",
            "--pinentry-mode", "loopback",
            "--passphrase", pass2,
            "--output", "decrypted_message.txt",
            "--decrypt", "message_for_agent2.txt.asc"
        ],
        capture_output=True,
        text=True
    )
    
    if decrypt_cmd.returncode != 0:
        print(f"Decryption failed: {decrypt_cmd.stderr}")
    assert decrypt_cmd.returncode == 0
    
    # Verify content
    with open("decrypted_message.txt", "r") as f:
        decrypted_content = f.read()
    
    print(f"Original: {message_content}")
    print(f"Decrypted: {decrypted_content}")
    
    assert decrypted_content == message_content
    print("Test completed successfully.")

if __name__ == "__main__":
    # Manual run setup without pytest dependency for direct execution
    temp_dir = tempfile.mkdtemp()
    try:
        os.environ["GNUPGHOME"] = temp_dir
        os.chmod(temp_dir, 0o700)
        
        # Create plain_message.txt
        with open("plain_message.txt", "w") as f:
            f.write("This is a plain text message.")
            
        test_encrypted_communication(temp_dir)
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir)
        # Cleanup files in CWD
        for f in ["plain_message.txt", "message_for_agent2.txt", "message_for_agent2.txt.asc", "decrypted_message.txt"]:
            if os.path.exists(f):
                os.remove(f)

