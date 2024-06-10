tic;
% choice of polarization, 'E' for transverse electric, 'M' for magnetic.
T = 'M';
% physical constants: Speed of light (Tmum/s), Planck¡¯s constant (eV/THz),
% and Vacuum permittivity (e^2/eV/mum). mu==1e-6, T==1e12.
Tc0 = 2.9979e2; hT = 4.1357e-3; e2epsilon0 = 55.2635;
hc = 1.2398; hBarT = 6.5821e-4; % precomputed relative constants
% physical parameters: permittivities, Fermi level position (eV),
% lateral period (mum), incident angle, normal vector,
% Fermi velocity (mum/s) and carrier lifetime (s).
epsilon_u = 3; epsilon_w = 4; E_F = 0.4;
D = [1,2,4,8]; theta = 0; N = [0;1]; vF = 1; tau = 0.09;
Gamma = 3.7e-3; % dirac constant multiplies relaxation rate (eV)
tau_u = (T=='E') + (T=='M') / epsilon_u;
tau_w = (T=='E') + (T=='M') / epsilon_w;
C2 = (T=='E') + (T=='M') * (epsilon_u/epsilon_w); % scaling factors
% numerical values: gridpoints number, identity matrix,
% perturbation truncation order and perturbation parameter.
Nx = 128; I = eye(Nx); L = 16; delta = 1;
P = [0:Nx/2-1 -Nx/2:-1]; P = P'; % Fourier wavenumber
Nf = 128; F = 1:12/Nf:13; % incident linear frequency (THz)
% row: 4 material periods; col: absorbance under different frequencies
Aloc = zeros(4,Nf); Anloc = zeros(4,Nf);
Acolor = [[0, 0.4470, 0.7410]; [0.8500, 0.3250, 0.0980];...
    [0.9290, 0.6940, 0.1250]; [0.4940, 0.1840, 0.5560]];
for j = 1:4
    d = D(j);
    x = 0:d/Nx:d-d/Nx; x = x'; % x=d omitted for periodicity
    % envelope function
    X0 = 1; w = d/2;
    X1 = @(x) -X0 + (d/2-w/2<x & x<d/2+w/2) .* sqrt(1-4*((x-d/2)/w).^2);
    for k = 1:Nf+1
        % equation coefficients
        f = F(k); omega = f*2*pi; % angular frequency
        k0 = omega / Tc0;
        ku = sqrt(epsilon_u) * k0;
        kw = sqrt(epsilon_w) * k0;
        alpha = ku * sin(theta);
        alpha_p = alpha + 2*pi/d * P;
        gamma_u = ku * cos(theta);
        gamma_w = kw * cos(theta);
        gamma_u_p = (alpha_p.^2 <= ku^2) .* sqrt(ku^2 - alpha_p.^2) ...
            + 1i * (alpha_p.^2 > ku^2) .* sqrt(alpha_p.^2 - ku^2);
        gamma_w_p = (alpha_p.^2 <= kw^2) .* sqrt(kw^2 - alpha_p.^2) ...
            + 1i * (alpha_p.^2 > kw^2) .* sqrt(alpha_p.^2 - kw^2);
        xi = -1 * ones(Nx,1); nu = 1i * gamma_u * ones(Nx,1);
        % model coefficients
        sigma0 = 2*E_F/(hc*e2epsilon0*(Gamma-1i*hBarT*omega));
        A0 = (T=='M') * norm(N) * sigma0 / (1i * k0);
        B0 = (T=='E') * 1i * k0 * sigma0 / norm(N); % constant part
        sigma_BGK = sigma0*vF^2*(3*f+2i/tau)/(4*f*(f+1i/tau)^2);
        A1 = (T=='M') * norm(N) * sigma_BGK / (1i * k0);
        B1 = (T=='E') * 1i * k0 * sigma_BGK / norm(N); % operater part
        % solve linear recursion
        U = zeros(Nx, L+1); U_hat = zeros(Nx, L+1);
        W = zeros(Nx, L+1); W_hat = zeros(Nx, L+1);
        G0 = diag(-1i*gamma_u_p); J0 = diag(-1i*gamma_w_p); % Fourier DNOs
        % local
        LHS = [I, -I+A0*X0*tau_w*J0; tau_u*G0, tau_w*J0-B0*X0*I];
        RHS1 = xi; RHS2 = -tau_u * nu;
        for ell = 1:L+1
            Rec = LHS \ [fft(RHS1);fft(RHS2)];
            U_hat(:,ell) = Rec(1:Nx); W_hat(:,ell) = Rec(Nx+1:2*Nx);
            U(:,ell) = ifft(U_hat(:,ell)); W(:,ell) = ifft(W_hat(:,ell));
            if ell < L+1
                V_hat = tau_w * J0 * W_hat(:,ell);
                RHS1 = -A0 * X1(x) .* ifft(V_hat);
                RHS2 = B0 * X1(x) .* W(:,ell);
            end
        end
        % add up all orders via Pade sum
        M = floor(L/2);
        U_hat_p = zeros(Nx,1); W_hat_p = zeros(Nx,1);
        for jj=1:Nx
            coeff = U_hat(jj,:).';
            U_hat_p(jj) = padesum(coeff,delta,M)/Nx;
            coeff = W_hat(jj,:).';
            W_hat_p(jj) = padesum(coeff,delta,M)/Nx;
        end
        % Absorbance spectra
        e_u_p = gamma_u_p .* (U_hat_p .* conj(U_hat_p)) / gamma_u_p(1);
        e_w_p = gamma_w_p .* (W_hat_p .* conj(W_hat_p)) / gamma_u_p(1);
        R = (alpha_p.^2 <= ku^2)' * e_u_p;
        S = (alpha_p.^2 <= kw^2)' * e_w_p;
        Aloc(j,k) = 1 - R - C2*S;
        % nonlocal
        LHS = [I, -I + A0*X0*tau_w*J0 + A1*X0*tau_w*diag(P.^2)*J0;...
            tau_u*G0, tau_w*J0 - B0*X0*I - B1*X0*diag(P.^2)];
        RHS1 = xi; RHS2 = -tau_u * nu;
        for ell = 1:L+1
            Rec = LHS \ [fft(RHS1);fft(RHS2)];
            U_hat(:,ell) = Rec(1:Nx); W_hat(:,ell) = Rec(Nx+1:2*Nx);
            U(:,ell) = ifft(U_hat(:,ell)); W(:,ell) = ifft(W_hat(:,ell));
            if ell < L+1
                V_hat = tau_w * J0 * W_hat(:,ell);
                RHS1 = -A0*X1(x).*ifft(V_hat) - A1*X1(x).*ifft(P.^2.*V_hat);
                RHS2 = B0*X1(x).* W(:,ell) + B1*X1(x).*ifft(P.^2.*W_hat(:,ell));
            end
        end
        M = floor(L/2);
        U_hat_p = zeros(Nx,1); W_hat_p = zeros(Nx,1);
        for jj=1:Nx
            coeff = U_hat(jj,:).';
            U_hat_p(jj) = padesum(coeff,delta,M)/Nx;
            coeff = W_hat(jj,:).';
            W_hat_p(jj) = padesum(coeff,delta,M)/Nx;
        end
        e_u_p = gamma_u_p .* (U_hat_p .* conj(U_hat_p)) / gamma_u_p(1);
        e_w_p = gamma_w_p .* (W_hat_p .* conj(W_hat_p)) / gamma_u_p(1);
        R = (alpha_p.^2 <= ku^2)' * e_u_p;
        S = (alpha_p.^2 <= kw^2)' * e_w_p;
        Anloc(j,k) = 1 - R - C2*S;
    end
    plot(F,Aloc(j,:),'--','Color',Acolor(j,:))
    hold on
    plot(F,Anloc(j,:),'Color',Acolor(j,:))
end
title('Absorbance vs. Incident Linear Frequency via HOPE')
xlabel('f (THz)')
ylabel('A')
legend('loc d = 1\mum','nloc d = 1\mum','loc d = 2\mum','nloc d = 2\mum',...
    'loc d = 4\mum','nloc d = 4\mum','loc d = 8\mum','nloc d = 8\mum')
hold off
saveas(gcf,'hope.png')
fprintf('HOPE terminates with %f seconds.\n',toc);