import paramiko
import os
import subprocess
HOSTNAME = '192.168.150.143'#'169.254.52.141'
USERNAME = 'pi'
PASSWORD = 'QuanserPi3'

if __name__ == "__main__":
    # Create SSH client
    ssh = paramiko.SSHClient()

    # Automatically add the host key to the local known_hosts file
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connect to the host
    ssh.connect(HOSTNAME, username=USERNAME, password=PASSWORD)

    # Execute a command on the host
    stdin, stdout, stderr = ssh.exec_command("cd Desktop/ && python ImageCapture.py")
    while stdout.read().decode() != "Capture Finished":
        pass
    command = "sshpass -p " + PASSWORD + " scp -r pi@" + HOSTNAME + ":/home/pi/Desktop/Images/ ./"
    os.system("sshpass -p 'QuanserPi3' scp -r pi@" + HOSTNAME + ":/home/pi/Desktop/Images/ ./")
    #stdin, stdout, stderr = ssh.exec_command('ls')

    # Print the output of the command
    print(stdout.read().decode())

    subprocess.run(['python', 'test.py'])
    subprocess.run(['python', 'SpotDetection.py'])

    # Close connection
    ssh.close()
