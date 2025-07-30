clear all
clc
close all
pkg load mapping % pode apresentar um erro de n�o carregar esse pacote, se apagar esta linha e colocar ela novamente o erro desaparece 

%Dados:

r=[10016.34 -17012.52 7899.28];
v=[2.5 -1.05 3.88];
u=3.986e5;
K=[0 0 1];

%Semi-eixo maior:

E=((norm(v))^2)/2-u/norm(r);
a=-u/(2*E);

%Excentricidade:

e=1/u*(((norm(v)^2)-u/norm(r))*r-(dot(r,v))*v);
norm(e);

%Ascensão reta do nodo ascendente:

h=cross(r,v);
N=cross(K,h);
Nj=N(2);
Ni=N(1);
O=acosd(Ni/norm(N));

if(Nj<0)
  O=360-O;
else
  O;
endif
O2=O;
%Inclinação:

hz=h(3);
i=acosd(hz/norm(h));

%Argumento do pericentro:

ek=e(3);
w=acosd((dot(N,e))/(norm(e)*norm(N)));

if(ek<0)
  w=360-w;
else
  w;
endif
w2=w;
%Anomalia verdadeira:

theta=acosd(dot(e,r)/(norm(e)*norm(r)));
rv=dot(r,v);

if(rv<0)
  theta=360-theta;
else
  theta;
endif

%Periodo:

T=sqrt(4*pi^2*a^3/u);

printf("Resultado Anal�tico:\n\nSemi-eixo maior: %d km\nExcentricidade: %d\nAscens�o reta do nodo ascendente: %d�\nInclina��o: %d�\nArgumento do pericentro: %d�\nAnomalia verdadeira: %d�\nPer�odo Orbital: %d segundos\n\n\n",a,norm(e),O,i,w,theta,T)

%Solução Númerica

x=[r(1) r(2) r(3) v(1) v(2) v(3)];

function [xp] = problema1 (t,x)

  u=3.986e5;
  r=[x(1) x(2) x(3)];

  xp(1)=x(4);
  xp(4)=(-u/(norm(r))^3)*x(1);
  xp(2)=x(5);
  xp(5)=(-u/(norm(r))^3)*x(2);
  xp(3)=x(6);
  xp(6)=(-u/(norm(r))^3)*x(3);

endfunction

opts=odeset("AbsTol",1e-6,"RelTol",1e-6);
[T,x]=ode45(@problema1,[0 T], [r(1);r(2);r(3);v(1);v(2);v(3)],opts );
p=0;

dimenT=numel(T);
while p<dimenT

p=p+1;

T(p);
%Dados:

r=[x(p,1) x(p,2) x(p,3)];
v=[x(p,4) x(p,5) x(p,6)];

%Semi-eixo maior:

E=((norm(v))^2)/2-u/norm(r);
a1(p,1)=-u/(2*E);

%Excentricidade:

e=1/u*(((norm(v)^2)-u/norm(r))*r-(dot(r,v))*v);
e1(p,1)=norm(e);

%Ascensão reta do nodo ascendente:

h=cross(r,v);
N=cross(K,h);
Nj=N(2);
Ni=N(1);
O=acosd(Ni/norm(N));

if(Nj<0)
  O1(p,1)=360-O;
else
  O1(p,1)=O;
endif

%Inclinação:

hz=h(3);
i1(p,1)=acosd(hz/norm(h));

%Argumento do pericentro:

ek=e(3);
w=acosd((dot(N,e))/(norm(e)*norm(N)));

if(ek<0)
  w1(p,1)=360-w;
else
  w1(p,1)=w;
endif

%Anomalia verdadeira:

theta=acosd(dot(e,r)/(norm(e)*norm(r)));
rv=dot(r,v);

if(rv<0)
  theta1(p,1)=360-theta;
else
  theta1(p,1)=theta;
endif

end

%Plotar os gráficos

E = wgs84Ellipsoid ("km");

[X,Y,Z]=ellipsoid(0,0,0,E.SemimajorAxis, E.SemimajorAxis, E.SemiminorAxis);
figure
plot3(x(:,1),x(:,2),x(:,3))
hold
surf(X,Y,Z)
axis equal

figure
plot(a1(:,1))
figure
plot(e1(:,1))
figure
plot(O1(:,1))
figure
plot(i1(:,1))
figure
plot(w1(:,1))
figure
plot(theta1(:,1))

%Matriz de Rotação

%senso e cossenos

si=sind(i);
ci=cosd(i);

sO=sind(O2);
cO=cosd(O2);

sw=sind(w2);
cw=cosd(w2);

%rotação eixo z com O(angulo de ascesão)

Rzi=[cO sO 0;-sO cO 0;0 0 1];

%rotação eixo x com i(angulo de inclinação)

Rx=[1 0 0; 0 ci si; 0 -si ci];

%rotação eixo x com i(angulo de inclinação)

Rzo=[cw sw 0;-sw cw 0;0 0 1];

%Matriz de rotação

R=Rzo*Rx*Rzi;


%Plote da solução analitica

p=0;

while p<dimenT

p=p+1;

r1(p)=a*(1-norm(e)^2)/(1+norm(e)*cosd(theta1(p,1)));
X=r1(p)*cosd(theta1(p,1));
Y=r1(p)*sind(theta1(p,1));
Z=0;

IN(1,1)=X;
IN(2,1)=Y;
IN(3,1)=Z;

Nv=R'*IN;

X1(p,1)=Nv(1);
Y1(p,1)=Nv(2);
Z1(p,1)=Nv(3);

end


figure
plot3(X1(:,1),Y1(:,1),Z1(:,1))
hold
plot3(x(:,1),x(:,2),x(:,3))
axis equal
