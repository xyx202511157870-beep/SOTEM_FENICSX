from sotem_ip import fit_cole_cole_debye


def main():
    fit = fit_cole_cole_debye(rho0=100.0, m=0.3, tau=1.0, c=0.3, n_terms=10)
    print(f"sigma_inf = {fit.sigma_infinity:.6e} S/m")
    print(f"relative L2 = {fit.relative_l2:.6e}")
    for i, term in enumerate(fit.terms):
        print(f"{i:02d}: delta_sigma={term.delta_sigma:.6e} tau={term.tau:.6e}")


if __name__ == "__main__":
    main()

