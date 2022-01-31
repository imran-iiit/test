
def count_ones(num):
    count = 0
    while num:
        if num & 1:
            count += 1
        num >>= 1
    return count

if __name__ == '__main__':
    num = 64
    print(f'Num bits in {num} --> {count_ones(num)}')
    assert count_ones(3) == 2
    assert count_ones(7) == 3
    assert count_ones(15) == 4
    assert count_ones(8) == 1
    assert count_ones(32) == 1