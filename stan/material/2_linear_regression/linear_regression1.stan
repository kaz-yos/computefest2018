data {
    int<lower=1> N;
    int<lower=1> M;
    matrix[M,N] X;
    real y[N];
}

parameters {
    vector[M] beta;
    real alpha;
    real<lower=0> sigma;
}

model {
    /* linear predictor */
    vector[N] mu = X' * beta + alpha;

    for (m in 1:M) {
        beta[m] ~ normal(0, 10);
    }
    alpha ~ normal(0, 10);
    sigma ~ normal(0, 5);

    for (n in 1:N) {
        y[n] ~ normal(mu[n], sigma);
    }
}

generated quantities {
    real y_ppc[N];
    /* Temporary scope, not kept */
    {
        vector[N] mu = X' * beta + alpha;
        for (n in 1:N) {
            y_ppc[n] = normal_rng(mu[n], sigma);
        }
    }
}
