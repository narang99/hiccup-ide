import subprocess


def aws_s3_sync(src: str, dest: str) -> int:
    process = subprocess.Popen(
        ["aws", "s3", "sync", src, dest],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in process.stdout:  # ty: ignore
        print(line, end="", flush=True)

    process.wait()
    return process.returncode
