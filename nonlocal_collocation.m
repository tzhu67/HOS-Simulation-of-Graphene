% see hops for detailed explanation
tic;
T = 'M';
Tc0 = 2.9979e2; hT = 4.1357e-3; e2epsilon0 = 55.2635;
hc = 1.2398; hBarT = 6.5821e-4;
epsilon_u = 3; epsilon_w = 4; E_F = 0.4;
D = [1,2,4,8]; theta = 0; N = [0;1]; vF = 1; tau = 0.09;
Gamma = 3.7e-3; delta = 1;
tau_u = (T=='E') + (T=='M') / epsilon_u;
tau_w = (T=='E') + (T=='M') / epsilon_w;
C2 = (T=='E') + (T=='M') * (epsilon_u/epsilon_w);
Nx = 128; I = eye(Nx);
P = [0:Nx/2-1 -Nx/2:-1]; P = P';
Nf = 128; F = 1:12/Nf:13;
Aloc = zeros(4,Nf); Anloc = zeros(4,Nf);
Acolor = [[0, 0.4470, 0.7410]; [0.8500, 0.3250, 0.0980];...
    [0.9290, 0.6940, 0.1250]; [0.4940, 0.1840, 0.5560]];
for j = 1:4
    d = D(j);
    x = 0:d/Nx:d-d/Nx; x = x';
    X0 = 1; w = d/2;
    X = @(x) X0 + delta * (-X0 + (d/2-w/2<x & x<d/2+w/2) .* sqrt(1-4*((x-d/2)/w).^2));
    for k = 1:Nf+1
        f = F(k); omega = f*2*pi;
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
        sigma0 = 2*E_F/(hc*e2epsilon0*(Gamma-1i*hBarT*omega));
        A0 = (T=='M') * norm(N) * sigma0 / (1i * k0);
        B0 = (T=='E') * 1i * k0 * sigma0 / norm(N);
        sigmaBGK = sigma0*vF^2*(3*f+2i/tau)/(4*f*(f+1i/tau)^2);
        A1 = (T=='M') * norm(N) * sigmaBGK / (1i * k0);
        B1 = (T=='E') * 1i * k0 * sigmaBGK / norm(N);
        % Fourier DNOs
        G0hat = diag(-1i*gamma_u_p); J0hat = diag(-1i*gamma_w_p);
        % Recover physical DNOs and differentiation matrix
        G0 = I; J0 = I; Diff = I;
        for i = 1:Nx
            G0(:,i) = ifft(G0hat * fft(I(:,i)));
            J0(:,i) = ifft(J0hat * fft(I(:,i)));
            Diff(:,i) = ifft(diag(1i*P) * fft(I(:,i)));
        end
        % solve linear system
        % local
        LHS = [I -I+A0*diag(X(x))*tau_w*J0;tau_u*G0 tau_w*J0-B0*diag(X(x))];
        RHS = [xi; -tau_u*nu];
        Sol = LHS \ RHS;
        U = Sol(1:Nx); W = Sol(Nx+1:2*Nx);
        U_hat = fft(U)/Nx; W_hat = fft(W)/Nx;
        e_u_p = gamma_u_p .* (U_hat .* conj(U_hat)) / gamma_u_p(1);
        e_w_p = gamma_w_p .* (W_hat .* conj(W_hat)) / gamma_u_p(1);
        R = (alpha_p.^2 <= ku^2)' * e_u_p;
        S = (alpha_p.^2 <= kw^2)' * e_w_p;
        Aloc(j,k) = 1 - R - C2*S;
        % nonlocal
        LHS = [I -I+(A0*I-A1*Diff^2)*diag(X(x))*tau_w*J0; ...
            tau_u*G0 tau_w*J0-(B0*I-B1*Diff^2)*diag(X(x))];
        Sol = LHS \ RHS;
        U = Sol(1:Nx); W = Sol(Nx+1:2*Nx);
        U_hat = fft(U)/Nx; W_hat = fft(W)/Nx;
        e_u_p = gamma_u_p .* (U_hat .* conj(U_hat)) / gamma_u_p(1);
        e_w_p = gamma_w_p .* (W_hat .* conj(W_hat)) / gamma_u_p(1);
        R = (alpha_p.^2 <= ku^2)' * e_u_p;
        S = (alpha_p.^2 <= kw^2)' * e_w_p;
        Anloc(j,k) = 1 - R - C2*S;
    end
    plot(F,Aloc(j,:),'--','Color',Acolor(j,:))
    hold on
    plot(F,Anloc(j,:),'Color',Acolor(j,:))
end
title('Absorbance vs. Incident Linear Frequency via Collocation')
xlabel('f (THz)')
ylabel('A')
legend('loc d = 1\mum','nloc d = 1\mum','loc d = 2\mum','nloc d = 2\mum',...
    'loc d = 4\mum','nloc d = 4\mum','loc d = 8\mum','nloc d = 8\mum')
hold off
saveas(gcf,'collocation.png')
fprintf('Collocation terminates with %f seconds.\n',toc);