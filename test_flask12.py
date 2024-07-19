import multiprocessing
import subprocess

def run_app1():
    subprocess.run(['python', 'mainprog.py'])

def run_app2():
    subprocess.run(['python', 'F:/Saif_Interview_Project/Resume_Parser/app.py'])

if __name__ == '__main__':
    # Create separate processes for each Flask app
    process1 = multiprocessing.Process(target=run_app1)
    process2 = multiprocessing.Process(target=run_app2)

    # Start the processes
    process1.start()
    process2.start()

    # Optionally, wait for processes to complete
    process1.join()
    process2.join()
