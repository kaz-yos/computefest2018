data {
    int N;
    int y[N];
}

transformed data {
    int U = 9;
}

parameters {
    real <lower=0, upper=1> theta;
    real <lower=0> lambda;
}

model {
    theta ~ beta(1,1);
    lambda ~ normal(0,5);

    for (n in 1:N) {
        real lpdf = poisson_lpmf(y[n] | lambda) - poisson_lcdf(U | lambda);
        if (y[n] == 0)
            target += log_mix(theta, 0, lpdf);
        else
            target += log(1 - theta) + lpdf;
    }
}

generated quantities {
    int y_ppc[N] = rep_array(0, N);

    for (n in 1:N) {
        if (!bernoulli_rng(theta)) {
            y_ppc[n] = U + 1;
            while (y_ppc[n] > U)
                y_ppc[n] = poisson_rng(lambda);
        }
    }
}
