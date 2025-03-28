import secrets
import string


def base62(n, length=8):
    alphabet = string.digits + string.ascii_letters
    base = len(alphabet)
    s = ""
    while n:
        n, r = divmod(n, base)
        s = alphabet[r] + s
    return s.rjust(length, "0")


def generate_base62_id():
    n = secrets.randbits(48)  # 48 bits of randomness
    return base62(n)


if __name__ == "__main__":
    for _ in range(10):
        print(generate_base62_id())
