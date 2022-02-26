//                   MFEM Ultraweak DPG acoustics example
//
// Compile with: make uw_dpg
//

//     - Δ p - ω^2 p = f̃ ,   in Ω
//                 p = p_0, on ∂Ω

// First Order System

//  ∇ p + i ω u = 0, in Ω
//  ∇⋅u + i ω p = f, in Ω
//           p = p_0, in ∂Ω
// where f:=f̃/(i ω) 

// UW-DPG:
// 
// p ∈ L^2(Ω), u ∈ (L^2(Ω))^dim 
// p̂ ∈ H^1/2(Ω), û ∈ H^-1/2(Ω)  
// -(p,  ∇⋅v) + i ω (u , v) + < p̂, v⋅n> = 0,      ∀ v ∈ H(div,Ω)      
// -(u , ∇ q) + i ω (p , q) + < û, q >  = (f,q)   ∀ q ∈ H^1(Ω)
//                                  p̂  = p_0     on ∂Ω 

// Note: 
// p̂ := p on Γ_h (skeleton)
// û := u on Γ_h  

// -------------------------------------------------------------
// |   |     p     |     u     |    p̂      |    û    |  RHS    |
// -------------------------------------------------------------
// | v | -(p, ∇⋅v) | i ω (u,v) | < p̂, v⋅n> |         |         |
// |   |           |           |           |         |         |
// | q | i ω (p,q) |-(u , ∇ q) |           | < û,q > |  (f,q)  |  

// where (q,v) ∈  H^1(Ω) × H(div,Ω) 

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double p_exact_r(const Vector &x);
double p_exact_i(const Vector &x);
void u_exact_r(const Vector &x, Vector & u);
void u_exact_i(const Vector &x, Vector & u);
double rhs_func_r(const Vector &x);
double rhs_func_i(const Vector &x);
void gradp_exact_r(const Vector &x, Vector &gradu);
void gradp_exact_i(const Vector &x, Vector &gradu);
double divu_exact_r(const Vector &x);
double divu_exact_i(const Vector &x);
double d2_exact_r(const Vector &x);
double d2_exact_i(const Vector &x);
double hatp_exact_r(const Vector & X);
double hatp_exact_i(const Vector & X);
void hatu_exact(const Vector & X, Vector & hatu);
void hatu_exact_r(const Vector & X, Vector & hatu);
void hatu_exact_i(const Vector & X, Vector & hatu);

int dim;
double omega;

int main(int argc, char *argv[])
{
   const char *mesh_file = "../../../data/inline-quad.mesh";
   int order = 1;
   int delta_order = 1;
   bool visualization = true;
   double rnum=1.0;
   int ref = 1;
   double theta = 0.0;
   bool adjoint_graph_norm = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree)");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&rnum, "-rnum", "--number_of_wavelenths",
                  "Number of wavelengths");      
   args.AddOption(&delta_order, "-do", "--delta_order",
                  "Order enrichment for DPG test space.");     
   args.AddOption(&theta, "-theta", "--theta",
                  "Theta parameter for AMR");                    
   args.AddOption(&adjoint_graph_norm, "-graph-norm", "--adjoint-graph-norm",
                  "-no-graph-norm", "--no-adjoint-graph-norm",
                  "Enable or disable Adjoint Graph Norm on the test space");                                
   args.AddOption(&ref, "-ref", "--serial_ref",
                  "Number of serial refinements.");                               
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   omega = 2.0 * M_PI * rnum;


   Mesh mesh(mesh_file, 1, 1);
   dim = mesh.Dimension();


   // Define spaces
   // L2 space for p
   FiniteElementCollection *p_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *p_fes = new FiniteElementSpace(&mesh,p_fec);

   // Vector L2 space for u 
   FiniteElementCollection *u_fec = new L2_FECollection(order-1,dim);
   FiniteElementSpace *u_fes = new FiniteElementSpace(&mesh,u_fec, dim); 

   // H^1/2 space for p̂  
   FiniteElementCollection * hatp_fec = new H1_Trace_FECollection(order,dim);
   FiniteElementSpace *hatp_fes = new FiniteElementSpace(&mesh,hatp_fec);

   // H^-1/2 space for û  
   FiniteElementCollection * hatu_fec = new RT_Trace_FECollection(order-1,dim);   
   FiniteElementSpace *hatu_fes = new FiniteElementSpace(&mesh,hatu_fec);

   // testspace fe collections
   int test_order = order+delta_order;
   FiniteElementCollection * q_fec = new H1_FECollection(test_order, dim);
   FiniteElementCollection * v_fec = new RT_FECollection(test_order-1, dim);


   mfem::out << "p_fes space true dofs = " << p_fes->GetTrueVSize() << endl;
   mfem::out << "u_fes space true dofs = " << u_fes->GetTrueVSize() << endl;
   mfem::out << "hadp_fes space true dofs = " << hatp_fes->GetTrueVSize() << endl;
   mfem::out << "hadu_fes space true dofs = " << hatu_fes->GetTrueVSize() << endl;


   // Coefficients
   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);
   Vector vec0(dim); vec0 = 0.;
   VectorConstantCoefficient vzero(vec0);
   ConstantCoefficient negone(-1.0);
   ConstantCoefficient omeg(omega);
   ConstantCoefficient omeg2(omega*omega);
   ConstantCoefficient negomeg(-omega);

   // Normal equation weak formulation
   Array<FiniteElementSpace * > trial_fes; 
   Array<FiniteElementCollection * > test_fec; 

   trial_fes.Append(p_fes);
   trial_fes.Append(u_fes);
   trial_fes.Append(hatp_fes);
   trial_fes.Append(hatu_fes);

   test_fec.Append(q_fec);
   test_fec.Append(v_fec);

   ComplexNormalEquations * a = new ComplexNormalEquations(trial_fes,test_fec);

   // i ω (p,q)
   a->AddTrialIntegrator(nullptr,new MixedScalarMassIntegrator(omeg),0,0);

// -(u , ∇ q)
   a->AddTrialIntegrator(new TransposeIntegrator(new GradientIntegrator(negone)),nullptr,1,0);

// -(p, ∇⋅v)
   a->AddTrialIntegrator(new MixedScalarWeakGradientIntegrator(one),nullptr,0,1);

//  i ω (u,v)
   a->AddTrialIntegrator(nullptr,new TransposeIntegrator(new VectorFEMassIntegrator(omeg)),1,1);

// < p̂, v⋅n>
   a->AddTrialIntegrator(new NormalTraceIntegrator,nullptr,2,1);

// < û,q >
   a->AddTrialIntegrator(new TraceIntegrator,nullptr,3,0);


// test integrators 

   //space-induced norm for H(div) × H1
   // (∇q,∇δq)
   a->AddTestIntegrator(new DiffusionIntegrator(one),nullptr,0,0);
   // (q,δq)
   a->AddTestIntegrator(new MassIntegrator(one),nullptr,0,0);
   // (∇⋅v,∇⋅δv)
   a->AddTestIntegrator(new DivDivIntegrator(one),nullptr,1,1);
   // (v,δv)
   a->AddTestIntegrator(new VectorFEMassIntegrator(one),nullptr,1,1);

   // additional integrators for the adjoint graph norm
   if (adjoint_graph_norm)
   {   
      // -i ω (∇q,δv)
      a->AddTestIntegrator(nullptr,new MixedVectorGradientIntegrator(negomeg),0,1);
      // i ω (v,∇ δq)
      a->AddTestIntegrator(nullptr,new MixedVectorWeakDivergenceIntegrator(negomeg),1,0);
      // ω^2 (v,δv)
      a->AddTestIntegrator(new VectorFEMassIntegrator(omeg2),nullptr,1,1);

      // - i ω (∇⋅v,δq)   
      a->AddTestIntegrator(nullptr,new VectorFEDivergenceIntegrator(negomeg),1,0);
      // i ω (q,∇⋅v)   
      a->AddTestIntegrator(nullptr,new MixedScalarWeakGradientIntegrator(negomeg),0,1);
      // ω^2 (q,δq)
      a->AddTestIntegrator(new MassIntegrator(omeg2),nullptr,0,0);
   }

   // RHS
   FunctionCoefficient f_rhs_r(rhs_func_r);
   FunctionCoefficient f_rhs_i(rhs_func_i);
   a->AddDomainLFIntegrator(new DomainLFIntegrator(f_rhs_r),new DomainLFIntegrator(f_rhs_i),0);
   a->Assemble();

   Array<int> ess_tdof_list;
   Array<int> ess_bdr;
   if (mesh.bdr_attributes.Size())
   {
      ess_bdr.SetSize(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      hatp_fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // shift the ess_tdofs
   for (int i = 0; i < ess_tdof_list.Size(); i++)
   {
      ess_tdof_list[i] += p_fes->GetTrueVSize() + u_fes->GetTrueVSize();
   }

   Array<int> offsets(5);
   offsets[0] = 0;
   offsets[1] = p_fes->GetVSize();
   offsets[2] = u_fes->GetVSize();
   offsets[3] = hatp_fes->GetVSize();
   offsets[4] = hatu_fes->GetVSize();
   offsets.PartialSum();
   BlockVector x_r(offsets); x_r = 0.0;
   BlockVector x_i(offsets); x_i = 0.0;

   FunctionCoefficient hatpex_r(hatp_exact_r);
   FunctionCoefficient hatpex_i(hatp_exact_i);
   GridFunction hatp_gf_r;
   GridFunction hatp_gf_i;

   hatp_gf_r.MakeRef(hatp_fes,x_r.GetBlock(2));
   hatp_gf_r.ProjectBdrCoefficient(hatpex_r,ess_bdr);

   hatp_gf_i.MakeRef(hatp_fes,x_i.GetBlock(2));
   hatp_gf_i.ProjectBdrCoefficient(hatpex_i,ess_bdr);

   OperatorPtr Ah_r, Ah_i;
   Vector X_r,B_r;
   Vector X_i,B_i;
   a->FormLinearSystem(ess_tdof_list,x_r,x_i,Ah_r, Ah_i,
                       X_r,X_i,B_r,B_i);


   // Ah_r.As<BlockMatrix>()->PrintMatlab();
   // Ah_i.As<BlockMatrix>()->PrintMatlab();

   // X_r.Print();
   // X_i.Print();
   // B_r.Print();
   // B_i.Print();

   // FunctionCoefficient pex(p_exact);
   // VectorFunctionCoefficient uex(dim,u_exact);


   delete a;
   delete q_fec;
   delete v_fec;
   delete hatp_fes;
   delete hatp_fec;
   delete hatu_fes;
   delete hatu_fec;
   delete u_fec;
   delete p_fec;
   delete u_fes;
   delete p_fes;

   return 0;
}

double p_exact_r(const Vector &x)
{
   return cos(omega*x.Sum());
}

double p_exact_i(const Vector &x)
{
   return sin(omega*x.Sum());
}

double hatp_exact_r(const Vector & X)
{
   return p_exact_r(X);
}

double hatp_exact_i(const Vector & X)
{
   return p_exact_i(X);
}

void gradp_exact_r(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   grad = -omega * sin(omega * x.Sum());
}

void gradp_exact_i(const Vector &x, Vector &grad)
{
   grad.SetSize(x.Size());
   grad = omega * cos(omega * x.Sum());
}

double d2_exact_r(const Vector &x)
{
   return -dim * omega * omega * cos(omega*x.Sum());
}

double d2_exact_i(const Vector &x)
{
   return -dim * omega * omega * sin(omega*x.Sum());
}

//  u = - ∇ p / (i ω )
//    = i (∇ p_r + i * ∇ p_i)  / ω 
//    = - ∇ p_i / ω + i ∇ p_r / ω 
void u_exact_r(const Vector &x, Vector & u)
{
   gradp_exact_i(x,u);
   u *= -1./omega;
}

void u_exact_i(const Vector &x, Vector & u)
{
   gradp_exact_r(x,u);
   u *= 1./omega;
}

void hatu_exact_r(const Vector & X, Vector & hatu)
{
   u_exact_r(X,hatu);
}
void hatu_exact_i(const Vector & X, Vector & hatu)
{
   u_exact_i(X,hatu);
}



//  ∇⋅u = i Δ p / ω
//      = i (Δ p_r + i * Δ p_i)  / ω 
//      = - Δ p_i / ω + i Δ p_r / ω 

double divu_exact_r(const Vector &x)
{
   return -d2_exact_i(x)/omega;
}

double divu_exact_i(const Vector &x)
{
   return d2_exact_r(x)/omega;
}


// f = ∇⋅u + i ω p 
// f_r = ∇⋅u_r - ω p_i  
double rhs_func_r(const Vector &x)
{
   double p = p_exact_i(x);
   double divu = divu_exact_r(x);
   return divu - omega * p;
}

// f_i = ∇⋅u_i + ω p_r
double rhs_func_i(const Vector &x)
{
   double p = p_exact_r(x);
   double divu = divu_exact_i(x);
   return divu + omega * p;
}







