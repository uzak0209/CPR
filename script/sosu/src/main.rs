fn main() {
    let n = 10_000_000;
    let wheel = [2, 3, 5, 7];
    let wheel_product: usize = wheel.iter().product();
    let wheel_size = 48;
    let increments = [
        2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6, 6, 2, 6, 4, 2, 6, 4, 6, 8, 4, 2, 4, 2,
        4, 8, 6, 4, 6, 2, 4, 6, 2, 6, 6, 4, 2, 4, 6, 2, 6, 4, 2, 4, 2, 10, 2, 10
    ];

    let sieve_size = (n - 1) / wheel_product + 1;
    let mut sieve = vec![true; sieve_size];
    let mut primes = wheel.to_vec();

    let mut i = 0;
    let mut num = 11;
    while num * num <= n {
        if sieve[i] {
            primes.push(num);
            let mut j = (num * num - 11) / wheel_product;
            while j < sieve_size {
                sieve[j] = false;
                j += num;
            }
        }
        num += increments[i % wheel_size];
        i += 1;
    }

    while num <= n {
        if i < sieve_size && sieve[i] {
            primes.push(num);
        }
        num += increments[i % wheel_size];
        i += 1;
    }

    println!("1000万以下の素数の数: {}", primes.len());
    println!("最初の10個の素数: {:?}", &primes[..10]);
    println!("最後の10個の素数: {:?}", &primes[primes.len() - 10..]);

    // 検証用: 1000以下の素数をすべて出力
    println!("1000以下の素数: {:?}", primes.iter().take_while(|&&x| x <= 1000).collect::<Vec<_>>());
}