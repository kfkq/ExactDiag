#include <iostream>
#include <vector>
#include <cmath>
#include <armadillo>
#include <unordered_map>
#include <set>
#include <fstream>
#include <iomanip>
#include <omp.h>

struct Bond {
    int i;
    int j;
    double Jz;
    double Jxy;
};

int main() {

    // Parameters
    // Read parameters from parameters.in
    int Lx, Ly, Nup;
    double J1_xy, J1_z, J2_xy, J2_z;
    double beta_start, beta_end;
    int N_beta;

    FILE* f = fopen("parameters.in", "r");
    if (!f) {
        std::cerr << "Error opening parameters.in" << std::endl;
        return 1;
    }
    fscanf(f, "%d", &Lx);
    fscanf(f, "%d", &Ly);
    fscanf(f, "%d", &Nup);
    fscanf(f, "%lf", &J1_xy);
    fscanf(f, "%lf", &J1_z);
    fscanf(f, "%lf", &J2_xy);
    fscanf(f, "%lf", &J2_z);
    fscanf(f, "%lf", &beta_start);
    fscanf(f, "%lf", &beta_end);
    fscanf(f, "%d", &N_beta);
    fclose(f);

    const int N = Lx * Ly;     // Total number of sites
    
    // Build bonds (nearest neighbor J1 and next-nearest neighbor J2)
    std::set<std::pair<int, int>> bond_set; // Used to avoid duplicates
    std::vector<Bond> bonds;
    
    auto add_bond = [&](int i, int j, double Jz_val, double Jxy_val) {
        int min_idx = std::min(i, j);
        int max_idx = std::max(i, j);
        if (min_idx == max_idx) return;
        if (bond_set.find({min_idx, max_idx}) == bond_set.end()) {
            bond_set.insert({min_idx, max_idx});
            Bond b;
            b.i = i;
            b.j = j;
            b.Jz = Jz_val;
            b.Jxy = Jxy_val;
            bonds.push_back(b);
        }
    };
    
    // Build nearest neighbor bonds
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            int idx = y*Lx + x;
            int right = y*Lx + (x+1)%Lx;
            int down = ((y+1)%Ly)*Lx + x;
            add_bond(idx, right, J1_z, J1_xy/2.0);
            add_bond(idx, down, J1_z, J1_xy/2.0);
        }
    }
    
    // Build next-nearest neighbor bonds only on even plaquettes
    for (int x = 0; x < Lx; x++) {
        for (int y = 0; y < Ly; y++) {
            // Check if plaquette is even
            if ((x+y) % 2 != 0) continue;
            
            // Add diagonals for this even plaquette
            int siteA = y*Lx + x;
            int siteC = ((y+1)%Ly)*Lx + (x+1)%Lx;
            add_bond(siteA, siteC, J2_z, J2_xy/2.0);
            
            int siteB = y*Lx + (x+1)%Lx;
            int siteD = ((y+1)%Ly)*Lx + x;
            add_bond(siteB, siteD, J2_z, J2_xy/2.0);
        }
    }
    
    // Generate basis for exactly Nup spin ups
    std::vector<arma::uword> basis;
    for (arma::uword s = 0; s < (arma::uword(1) << N); s++) {
        int count = 0;
        for (int b = 0; b < N; b++) {
            if (s & (arma::uword(1) << b)) 
                count++;
        }
        if (count == Nup)
            basis.push_back(s);
    }

    if (basis.empty()) {
        std::cerr << "No states with Nup = " << Nup << std::endl;
        return 1;
    }

    double Z = 0.0, E_total = 0.0, E2_total = 0.0;
    const arma::uword n = basis.size();
    std::cout << "Total basis states for diagonalization: " << n << std::endl;
    arma::mat H(n, n, arma::fill::zeros);
    std::unordered_map<arma::uword, arma::uword> stoi;
        
    for (arma::uword i = 0; i < n; i++)
        stoi[basis[i]] = i;
        
    // Parallel Hamiltonian construction
    #pragma omp parallel for
    for (arma::uword idx = 0; idx < n; idx++) {
        double diag = 0.0;
        const arma::uword state = basis[idx];
            
        // Compute diagonal terms: loop over all bonds
        for (const Bond& bond : bonds) {
            bool bit_i = (state >> bond.i) & 1;
            bool bit_j = (state >> bond.j) & 1;
            double sz_i = bit_i ? 0.5 : -0.5;
            double sz_j = bit_j ? 0.5 : -0.5;
            diag += bond.Jz * sz_i * sz_j;
        }
            
        H(idx, idx) = diag;
            
        // Off-diagonal terms: flip when spins are antiparallel
        for (const Bond& bond : bonds) {
            bool bit_i = (state >> bond.i) & 1;
            bool bit_j = (state >> bond.j) & 1;
                
            // If spins are aligned, skip
            if (bit_i == bit_j) continue;
                
            arma::uword new_state = state;
            // Flip both sites
            new_state ^= (1u << bond.i);
            new_state ^= (1u << bond.j);
            auto it = stoi.find(new_state);
            if (it != stoi.end()) {
                arma::uword j = it->second;
                // Only update lower triangle
                if (j < idx)
                    H(idx, j) += bond.Jxy;  // bond.Jxy is already Jxy/2
            }
        }
    }
        
    // Symmetrize Hamiltonian
    H = arma::symmatl(H);
        
    // Diagonalize
    arma::vec eigenvalues;
    arma::mat eigenvectors;
    arma::eig_sym(eigenvalues, eigenvectors, H);
        
    // Prepare for temperature loop
    std::ofstream out("output.dat");
    if (!out) {
        std::cerr << "Error opening output.dat" << std::endl;
        return 1;
    }

    out << "#" << std::left << std::setw(15) << " Temperature" 
        << std::setw(15) << "Energy" 
        << std::setw(15) << "SpecificHeat" << std::endl;
    
    for (int i = 0; i < N_beta; i++) {
        double beta = beta_start + i * (beta_end - beta_start) / (N_beta - 1.0);
        double Z = 0.0;
        double E_total = 0.0;
        double E2_total = 0.0;

        for (double E : eigenvalues) {
            double boltzmann = std::exp(-beta * E);
            Z += boltzmann;
            E_total += E * boltzmann;
            E2_total += E * E * boltzmann;
        }

        double avg_E = E_total / Z;
        double avg_E2 = E2_total / Z;
        double specific_heat = beta * beta * (avg_E2 - avg_E * avg_E);
        double temperature = 1.0 / beta;

        out << std::right << std::fixed << std::setprecision(6);
        out << std::setw(15) << temperature 
            << std::setw(15) << avg_E 
            << std::setw(15) << specific_heat << std::endl;
    }
    
    return 0;
}
