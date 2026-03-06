def compute_prime_sum(n):
    """
    计算小于 n 的所有质数之和
    
    参数:
        n: 整数，上限值（不包含）
    
    返回:
        小于 n 的所有质数之和
    """
    if n <= 2:
        return 0
    
    # 使用埃拉托斯特尼筛法（Sieve of Eratosthenes）找出所有质数
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            # 将 i 的所有倍数标记为非质数
            for j in range(i * i, n, i):
                is_prime[j] = False
    
    # 计算所有质数之和
    prime_sum = sum(i for i in range(n) if is_prime[i])
    
    return prime_sum


if __name__ == "__main__":
    # 测试示例
    test_cases = [10, 20, 100]
    for num in test_cases:
        result = compute_prime_sum(num)
        print(f"小于 {num} 的所有质数之和: {result}")
