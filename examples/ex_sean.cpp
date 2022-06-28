//                                MFEM LOR Example
//
// Compile with: make ex_sean
//
// Sample runs:
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ex9 -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.02 -s 13 -tf 9
//    ex9 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/star-mixed.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ex9 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
//    ex9 -m ../data/periodic-square.msh -p 0 -r 2 -dt 0.005 -tf 2
//    ex9 -m ../data/periodic-cube.msh -p 0 -r 1 -o 2 -tf 2
//
// Device sample runs:
//    ex9 -pa
//    ex9 -ea
//    ex9 -fa
//    ex9 -pa -m ../data/periodic-cube.mesh
//    ex9 -pa -m ../data/periodic-cube.mesh -d cuda
//    ex9 -ea -m ../data/periodic-cube.mesh -d cuda
//    ex9 -fa -m ../data/periodic-cube.mesh -d cuda
//    ex9 -pa -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9 -d cuda
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. The saving of time-dependent data files for external
//               visualization with VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) is also illustrated.

#include "mfem.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Node transform Function
void NodeShift(const IntegrationPoint &ip, const int &s, Vector &ip_trans);

// Calculate the "new" high order solution based on LOR
void CalculateLORInterp(GridFunction &u_HO_interp, const Mesh &mesh,
                        const Mesh &mesh_LOR, const GridFunction &x,
                        const GridFunction &u_LOR, GridFunction &node_vals,
                        FindPointsGSLIB &finder, const FiniteElementSpace &fes);
/*
// Calculate the Low Order Refined Solution
void CalculateLORSolution(GridTransfer &gt,
                          const GridFunction &u_HO,
                          GridFunction &u_LOR);
*/
// Calculate the Low Order Solution
void CalculateLOSolution(const GridFunction &u_HO, const GridFunction &x,
                         const double &dt, GridFunction &u_LO, Vector &el_mass,
                         Vector &el_vol);

// Function to calculate the mins and maxes of elements
void ComputeElementsMinMax(const Vector &u, const FiniteElementSpace &fes,
                           Vector &u_min, Vector &u_max, Array<bool> *active_el,
                           Array<bool> *active_dof);

// Updating time step
void UpdateTimeStepEstimate(const Vector &x, const Vector &dx,
                            const Vector &x_min, const Vector &x_max,
                            double &dt_est);

// For visualization, taken from lor-transfer.cpp
int Wx = 0, Wy = 0;                // window position
int Ww = 350, Wh = 350;            // window size
int offx = Ww + 5, offy = Wh + 25; // window offset

void visualize(VisItDataCollection &, string, int, int);

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

// FCT_Project
// Solves M xy = m, so that y_min <= xy_i / x_i <= y_max.
// @Sean - take x to be a vector of ones....
void FCT_Project(DenseMatrix &M, DenseMatrixInverse &M_inv, Vector &m,
                 Vector &x, double y_min, double y_max, Vector &xy);

class DG_Solver : public Solver {
private:
  SparseMatrix &M, &K, A;
  GMRESSolver linear_solver;
  BlockILU prec;
  double dt;

public:
  DG_Solver(SparseMatrix &M_, SparseMatrix &K_, const FiniteElementSpace &fes)
      : M(M_), K(K_), prec(fes.GetFE(0)->GetDof(),
                           BlockILU::Reordering::MINIMUM_DISCARDED_FILL),
        dt(-1.0) {
    linear_solver.iterative_mode = false;
    linear_solver.SetRelTol(1e-9);
    linear_solver.SetAbsTol(0.0);
    linear_solver.SetMaxIter(100);
    linear_solver.SetPrintLevel(0);
    linear_solver.SetPreconditioner(prec);
  }

  void SetTimeStep(double dt_) {
    if (dt_ != dt) {
      dt = dt_;
      // Form operator A = M - dt*K
      A = K;
      A *= -dt;
      A += M;

      // this will also call SetOperator on the preconditioner
      linear_solver.SetOperator(A);
    }
  }

  void SetOperator(const Operator &op) { linear_solver.SetOperator(op); }

  virtual void Mult(const Vector &x, Vector &y) const {
    linear_solver.Mult(x, y);
  }
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator {
private:
  BilinearForm &M, &K;
  const Vector &b;
  Solver *M_prec;
  CGSolver M_solver;
  DG_Solver *dg_solver;

  mutable Vector z;

public:
  FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_);

  virtual void Mult(const Vector &x, Vector &y) const;
  virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

  virtual ~FE_Evolution();
};

int main(int argc, char *argv[]) {
  // 1. Parse command-line options.
  problem = 0;
  const char *mesh_file = "../data/periodic-square.mesh";
  int ref_levels = 1;
  int order = 2;
  bool pa = false;
  bool ea = false;
  bool fa = false;
  const char *device_config = "cpu";
  int ode_solver_type = 1;
  double t_final = 1.0;
  double dt = 0.0001;
  bool visualization = true;
  bool visit = false;
  bool paraview = false;
  bool binary = false;
  int vis_steps = 5;

  // Here we have a new variable for the cell averaging.
  // averaging = 0; No cell averaging, just high order solution
  //             1; Low order cell averaging
  //             2; Low order refined (LOR)
  //             3; Both LO solution and LOR solution
  int averaging = 3;
  // LOR refinement level
  int lref = 2;

  int precision = 8;
  cout.precision(precision);

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
  args.AddOption(&problem, "-p", "--problem",
                 "Problem setup to use. See options in velocity_function().");
  args.AddOption(&ref_levels, "-r", "--refine",
                 "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order",
                 "Order (degree) of the finite elements.");
  args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                 "--no-partial-assembly", "Enable Partial Assembly.");
  args.AddOption(&ea, "-ea", "--element-assembly", "-no-ea",
                 "--no-element-assembly", "Enable Element Assembly.");
  args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa", "--no-full-assembly",
                 "Enable Full Assembly.");
  args.AddOption(&device_config, "-d", "--device",
                 "Device configuration string, see Device::Configure().");
  args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                 "ODE solver: 1 - Forward Euler,\n\t"
                 "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                 "            11 - Backward Euler,\n\t"
                 "            12 - SDIRK23 (L-stable), 13 - SDIRK33,\n\t"
                 "            22 - Implicit Midpoint Method,\n\t"
                 "            23 - SDIRK23 (A-stable), 24 - SDIRK34");
  args.AddOption(&t_final, "-tf", "--t-final", "Final time; start time is 0.");
  args.AddOption(&dt, "-dt", "--time-step", "Time step.");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                 "--no-visit-datafiles",
                 "Save data files for VisIt (visit.llnl.gov) visualization.");
  args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                 "--no-paraview-datafiles",
                 "Save data files for ParaView (paraview.org) visualization.");
  args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                 "--ascii-datafiles",
                 "Use binary (Sidre) or ascii format for VisIt data files.");
  args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                 "Visualize every n-th timestep.");
  args.AddOption(&averaging, "-avgg", "--averaging",
                 "Level of cell averaging.");
  args.AddOption(&lref, "-lref", "--lor-ref-level", "LOR refinement level.");
  args.Parse();
  if (!args.Good()) {
    args.PrintUsage(cout);
    return 1;
  }
  args.PrintOptions(cout);

  Device device(device_config);
  device.Print();

  // 2. Read the mesh from the given mesh file. We can handle geometrically
  //    periodic meshes in this code.
  Mesh mesh(mesh_file, 1, 1);
  int dim = mesh.Dimension();

  // 3. Define the ODE solver used for time integration. Several explicit
  //    Runge-Kutta methods are available.
  ODESolver *ode_solver = NULL;
  switch (ode_solver_type) {
  // Explicit methods
  case 1:
    ode_solver = new ForwardEulerSolver;
    break;
  case 2:
    ode_solver = new RK2Solver(1.0);
    break;
  case 3:
    ode_solver = new RK3SSPSolver;
    break;
  case 4:
    ode_solver = new RK4Solver;
    break;
  case 6:
    ode_solver = new RK6Solver;
    break;
  // Implicit (L-stable) methods
  case 11:
    ode_solver = new BackwardEulerSolver;
    break;
  case 12:
    ode_solver = new SDIRK23Solver(2);
    break;
  case 13:
    ode_solver = new SDIRK33Solver;
    break;
  // Implicit A-stable methods (not L-stable)
  case 22:
    ode_solver = new ImplicitMidpointSolver;
    break;
  case 23:
    ode_solver = new SDIRK23Solver;
    break;
  case 24:
    ode_solver = new SDIRK34Solver;
    break;

  default:
    cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
    return 3;
  }

  // 4. Refine the mesh to increase the resolution. In this example we do
  //    'ref_levels' of uniform refinement, where 'ref_levels' is a
  //    command-line parameter. If the mesh is of NURBS type, we convert it to
  //    a (piecewise-polynomial) high-order mesh.
  for (int lev = 0; lev < ref_levels; lev++) {
    mesh.UniformRefinement();
  }

  // Create the low-order refined mesh
  int basis_LOR = BasisType::ClosedUniform;
  Mesh mesh_LOR = Mesh::MakeRefined(mesh, lref, basis_LOR);

  if (mesh.NURBSext) {
    mesh.SetCurvature(max(order, 1));
    mesh_LOR.SetCurvature(max(order, 1));
  }
  mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));
  mesh_LOR.GetBoundingBox(bb_min, bb_max, max(order, 1));

  // 5. Define the discontinuous DG finite element space of the given
  //    polynomial order on the refined mesh.
  DG_FECollection fec(order, dim, BasisType::Positive);
  FiniteElementSpace fes(&mesh, &fec);

  // Discontinuous FE space for LOR
  int LOR_order = 0;
  DG_FECollection fec_LOR(LOR_order, dim, BasisType::Positive);
  FiniteElementSpace fes_LOR(&mesh_LOR, &fec_LOR);

  cout << "Number of unknowns: " << fes.GetVSize() << endl;

  // 6. Set up and assemble the bilinear and linear forms corresponding to the
  //    DG discretization. The DGTraceIntegrator involves integrals over mesh
  //    interior faces.
  VectorFunctionCoefficient velocity(dim, velocity_function);
  FunctionCoefficient inflow(inflow_function);
  FunctionCoefficient u0(u0_function);

  BilinearForm m(&fes);
  BilinearForm k(&fes);

  if (pa) {
    m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    k.SetAssemblyLevel(AssemblyLevel::PARTIAL);
  } else if (ea) {
    m.SetAssemblyLevel(AssemblyLevel::ELEMENT);
    k.SetAssemblyLevel(AssemblyLevel::ELEMENT);
  } else if (fa) {
    m.SetAssemblyLevel(AssemblyLevel::FULL);
    k.SetAssemblyLevel(AssemblyLevel::FULL);
  }

  m.AddDomainIntegrator(new MassIntegrator);

  constexpr double alpha = -1.0;
  k.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
  k.AddInteriorFaceIntegrator(
      new NonconservativeDGTraceIntegrator(velocity, alpha));
  k.AddBdrFaceIntegrator(new NonconservativeDGTraceIntegrator(velocity, alpha));

  LinearForm b(&fes);
  b.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(inflow, velocity, alpha));

  m.Assemble();

  int skip_zeros = 0;
  k.Assemble(skip_zeros);
  b.Assemble();
  m.Finalize();
  k.Finalize(skip_zeros);
  ;

  // 7. Define the initial conditions, save the corresponding grid function to
  //    a file and (optionally) save data in the VisIt format and initialize
  //    GLVis visualization.
  GridFunction u_HO(&fes);
  u_HO.ProjectCoefficient(u0);

  // Visualization
  VisItDataCollection HO_dc("HO", &mesh);
  HO_dc.RegisterField("Density", &u_HO);

  // Defining everything needed for the LO solution
  GridFunction u_LO(u_HO);
  VisItDataCollection LO_dc("LO", &mesh);
  LO_dc.RegisterField("Density", &u_LO);

  // Defining everything for the LOR solution
  GridFunction u_LOR(&fes_LOR);
  VisItDataCollection LOR_dc("LOR", &mesh_LOR);
  LOR_dc.RegisterField("Density", &u_LOR);

  // Print out meshes and initial solution
  {
    ofstream omesh("ex_sean.mesh");
    omesh.precision(precision);
    mesh.Print(omesh);
    ofstream osol("ex_sean-init.gf");
    osol.precision(precision);
    u_HO.Save(osol);

    // We only want to output when specified
    if (averaging == 2 || averaging == 3) {
      ofstream omesh_LOR("ex_sean_LOR.mesh");
      omesh_LOR.precision(precision);
      mesh_LOR.Print(omesh_LOR);

      ofstream osol_LOR("ex_sean_LOR-init.gf");
      osol_LOR.precision(precision);
      u_LOR.Save(osol_LOR);
    }
  }

  // Create data collection for solution output: either VisItDataCollection for
  // ascii data files, or SidreDataCollection for binary data files.
  DataCollection *dc_HO = NULL;
  DataCollection *dc_LO = NULL;
  DataCollection *dc_LOR = NULL;
  if (visit) {
    if (binary) {
#ifdef MFEM_USE_SIDRE
      dc_HO = new SidreDataCollection("Example_sean_HO", &mesh);
      dc_LO = new SidreDataCollection("Example_sean_LO", &mesh);
      dc_LOR = new SidreDataCollection("Example_sean_LOR", &mesh);
#else
      MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
    } else {
      dc_HO = new VisItDataCollection("Example_sean_HO", &mesh);
      dc_HO->SetPrecision(precision);

      dc_LO = new VisItDataCollection("Example_sean_LO", &mesh);
      dc_LO->SetPrecision(precision);

      dc_LOR = new VisItDataCollection("Example_sean_LOR", &mesh_LOR);
      dc_LOR->SetPrecision(precision);
    }
    dc_HO->RegisterField("solution_HO", &u_HO);
    dc_HO->SetCycle(0);
    dc_HO->SetTime(0.0);
    dc_HO->Save();

    dc_LO->RegisterField("solution_LO", &u_LO);
    dc_LO->SetCycle(0);
    dc_LO->SetTime(0.0);
    dc_LO->Save();

    dc_LOR->RegisterField("solution_LOR", &u_LOR);
    dc_LOR->SetCycle(0);
    dc_LOR->SetTime(0.0);
    dc_LOR->Save();
  }

  ParaViewDataCollection *pd_HO = NULL;
  ParaViewDataCollection *pd_LO = NULL;
  ParaViewDataCollection *pd_LOR = NULL;
  if (paraview) {
    pd_HO = new ParaViewDataCollection("Example_sean_HO", &mesh);
    pd_HO->SetPrefixPath("ParaView");
    pd_HO->RegisterField("solution_HO", &u_HO);
    pd_HO->SetLevelsOfDetail(order);
    pd_HO->SetDataFormat(VTKFormat::BINARY);
    pd_HO->SetHighOrderOutput(true);
    pd_HO->SetCycle(0);
    pd_HO->SetTime(0.0);
    pd_HO->Save();

    pd_LO = new ParaViewDataCollection("Example_sean_LO", &mesh);
    pd_LO->SetPrefixPath("ParaView");
    pd_LO->RegisterField("solution_LO", &u_LO);
    pd_LO->SetLevelsOfDetail(order);
    pd_LO->SetDataFormat(VTKFormat::BINARY);
    pd_LO->SetHighOrderOutput(true);
    pd_LO->SetCycle(0);
    pd_LO->SetTime(0.0);
    pd_LO->Save();

    pd_LOR = new ParaViewDataCollection("Example_sean_LOR", &mesh_LOR);
    pd_LOR->SetPrefixPath("ParaView");
    pd_LOR->RegisterField("solution_LOR", &u_LOR);
    pd_LOR->SetLevelsOfDetail(order);
    pd_LOR->SetDataFormat(VTKFormat::BINARY);
    pd_LOR->SetHighOrderOutput(true);
    pd_LOR->SetCycle(0);
    pd_LOR->SetTime(0.0);
    pd_LOR->Save();
  }

  // 8. Define the time-dependent evolution operator describing the ODE
  //    right-hand side, and perform time-integration (looping over the time
  //    iterations, ti, with a time-step dt).
  FE_Evolution adv(m, k, b);

  double t = 0.0;
  // double dt_est = dt;
  adv.SetTime(t);
  ode_solver->Init(adv);

  // Need to declare these variables for calculating u_LO
  Mesh *mesh_temp = fes.GetMesh();
  GridFunction x(mesh_temp->GetNodes()->FESpace());
  mesh_temp->GetNodes(x);

  // Projection onto the LOR space
  GridTransfer *gt;
  gt = new L2ProjectionGridTransfer(fes, fes_LOR);

  const Operator &R = gt->ForwardOperator();

  // Declaring vectors for the mass and volume
  const int NE = x.FESpace()->GetNE();
  Vector el_mass(NE);
  Vector el_vol(NE);
  Vector el_min(NE);
  Vector el_max(NE);

  // Here we do the interpolation to recover a smoother solution using
  // the LOR solution.
  FindPointsGSLIB finder;
  finder.Setup(mesh_LOR);
  GridFunction node_vals(x.FESpace());

  GridFunction u_HO_interp(&fes);
  VisItDataCollection HO_interp_dc("HO_interp", &mesh);
  HO_interp_dc.RegisterField("Density", &u_HO_interp);

  const Operator &P = gt->BackwardOperator();

  // Time loop for solving
  bool done = false;
  for (int ti = 0; !done;) {
    double dt_real = min(dt, t_final - t);
    ode_solver->Step(u_HO, t, dt_real);

    // Only compute this if specified
    if (averaging == 1 || averaging == 3) {
      // Calculate the LO solution using u_HO
      CalculateLOSolution(u_HO, x, dt_real, u_LO, el_mass, el_vol);
    }

    if (averaging == 2 || averaging == 3) {
      // Here we do an L2 projection onto the finer fes, gives u_LOR
      R.Mult(u_HO, u_LOR);
      // P.Mult(u_LOR, u_HO_interp);
    }

    ti++;

    done = (t >= t_final - 1e-8 * dt);

    if (done || ti % vis_steps == 0) {
      cout << "time step: " << ti << ", time: " << t << endl;

      if (visit) {
        dc_HO->SetCycle(ti);
        dc_HO->SetTime(t);
        dc_HO->Save();

        dc_LO->SetCycle(ti);
        dc_LO->SetTime(t);
        dc_LO->Save();
      }

      if (paraview) {
        pd_HO->SetCycle(ti);
        pd_HO->SetTime(t);
        pd_HO->Save();

        dc_LO->SetCycle(ti);
        dc_LO->SetTime(t);
        dc_LO->Save();
      }
    }
  }

  //---------------------------------------------------------------------------
  // Trying to send the LOR solution to the HO the chad L2 projection way

  Mesh *mesh_temp_LOR = fes_LOR.GetMesh();
  GridFunction x_LOR(mesh_temp_LOR->GetNodes()->FESpace());
  mesh_temp_LOR->GetNodes(x_LOR);

  auto *Tr_LOR = x_LOR.FESpace()->GetMesh()->GetElementTransformation(0);
  const int NE_LOR = x_LOR.FESpace()->GetNE();
  // Need this alue because it's how many subcells there are to a cell.
  const int subcell_num = lref * lref;

  const FiniteElement *fe_LOR = u_LOR.FESpace()->GetFE(0);
  const IntegrationRule &ir_LOR =
      MassIntegrator::GetRule(*fe_LOR, *fe_LOR, *Tr_LOR);
  const int nqp_LOR = ir_LOR.GetNPoints();
  // cout << "ir_LOR(0) = " << ir_LOR->IntPoint(0) << endl;

  // This creates a matrix of basis functions evaluated at
  // quadrature points.  Storage is nqpt x ndofs
  const DofToQuad &maps = fe_LOR->GetDofToQuad(ir_LOR, DofToQuad::TENSOR);
  cout << maps.B[0] << endl;

  // Grabbing information from the quadrature
  GeometricFactors geom_LOR(x_LOR, ir_LOR, GeometricFactors::DETERMINANTS);
  auto qi_u_LOR = u_LOR.FESpace()->GetQuadratureInterpolator(ir_LOR);
  Vector u_LOR_qvals(subcell_num * nqp_LOR * NE);

  qi_u_LOR->Values(u_LOR, u_LOR_qvals);

  Vector m_proj(nqp_LOR * NE);
  Vector ip_trans(3);

  // Integration loop to calculate main looping

  for (int k = 0; k < NE; k++) {
    for (int s = 0; s < subcell_num; s++) {
      for (int q = 0; q < nqp_LOR; q++) {
        const IntegrationPoint &ip_LOR = ir_LOR.IntPoint(q);
        // cout << "x coord of ir_LOR = " << ip_LOR.x << endl;
        // cout << "y coord of ir_LOR = " << ip_LOR.y << endl;
        // cout << endl;
        NodeShift(ip_LOR, s, ip_trans);
        // cout << "x coord of new ir_LOR = " << ip_trans(0) << endl;
        // cout << "y coord of new ir_LOR = " << ip_trans(1) << endl;
        // cout << endl;

        // The indexing is a little confusing but this should be correct
        // Need to map these values into the HO space
        // The nods on the reference element here are smol compared to the
        // macro element, so we need a function to essentially map
        // [0,1] to [0,.5]
        m_proj(k) +=
            ip_LOR.weight *
            geom_LOR.detJ(k * subcell_num * nqp_LOR + s * nqp_LOR + q) *
            u_LOR_qvals(k * subcell_num + s);
      }
    }
  }

  //---------------------------------------------------------------------------

  // Here we do the interpolation to recover a smoother solution using
  // the LOR solution.
  CalculateLORInterp(u_HO_interp, mesh, mesh_LOR, x, u_LOR, node_vals, finder,
                     fes);

  // 9. Save the final solution. This output can be viewed later using GLVis:
  if (visualization) {
    visualize(HO_dc, "HO", Wx, Wy);
    Wx += offx;
    if (averaging == 1 || averaging == 3) {
      visualize(LO_dc, "LO", Wx, Wy);
      Wx += offx;
    }

    if (averaging == 2 || averaging == 3) {
      visualize(LOR_dc, "LOR", Wx, Wy);
      Wx += offx;
    }
    visualize(HO_interp_dc, "HO_interp", Wx, Wy);
    Wx += offx;
  }

  {
    ofstream osol_HO("ex_sean-final_HO.gf");
    osol_HO.precision(precision);
    u_HO.Save(osol_HO);

    ofstream osol_HO_interp("ex_sean-final_HO_interp.gf");
    osol_HO_interp.precision(precision);
    u_HO_interp.Save(osol_HO_interp);

    if (averaging == 1 || averaging == 3) {
      ofstream osol_LO("ex_sean-final_LO.gf");
      osol_LO.precision(precision);
      u_LO.Save(osol_LO);
    }

    if (averaging == 2 || averaging == 3) {
      ofstream osol_LOR("ex_sean-final_LOR.gf");
      osol_LOR.precision(precision);
      u_LOR.Save(osol_LOR);
    }
  }

  // 10. Free the used memory.
  delete ode_solver;
  delete pd_LO;
  delete pd_LOR;
  delete pd_HO;
  delete dc_LO;
  delete dc_HO;
  delete dc_LOR;

  return 0;
}

// Need this function to shift the reference node values for the L2 Projection
void NodeShift(const IntegrationPoint &ip, const int &s, Vector &ip_trans) {
  Vector temp(3);
  ip_trans(0) = ip.x;
  ip_trans(1) = ip.y;
  ip_trans(2) = 1;

  DenseMatrix trans(3);
  trans(0, 0) = .5;
  trans(1, 1) = .5;
  trans(2, 2) = 1;

  if (s == 1) {
    trans(0, 2) = .5;
  } else if (s == 2) {
    trans(1, 2) = .5;
  } else if (s == 3) {
    trans(0, 2) = .5;
    trans(1, 2) = .5;
  }

  trans.Mult(ip_trans, temp);

  ip_trans = temp;
}

void CalculateLORInterp(GridFunction &u_HO_interp, const Mesh &mesh,
                        const Mesh &mesh_LOR, const GridFunction &x,
                        const GridFunction &u_LOR, GridFunction &node_vals,
                        FindPointsGSLIB &finder,
                        const FiniteElementSpace &fes) {
  const int NE = x.FESpace()->GetNE();
  int dim = mesh.Dimension();

  // This does not give us what we want I think
  mesh.GetNodes(node_vals);

  // Grabbing information from the finite element space
  const FiniteElement *zone = fes.GetFE(0);
  int num_ldofs = zone->GetDof();

  IntegrationRule zone_dofs(num_ldofs);

  // Output number of dofs per cell
  cout << "Number of dofs per cell: " << num_ldofs << endl;
  for (int i = 0; i < num_ldofs; ++i) {
    zone_dofs[i] = zone->GetNodes()[i];
  }

  // Element restriction
  const Operator *x_elem_restrict_lex =
      x.FESpace()->GetElementRestriction(ElementDofOrdering::LEXICOGRAPHIC);

  Vector x_local;
  x_local.SetSize(x_elem_restrict_lex->Height());
  x_elem_restrict_lex->Mult(x, x_local);

  // Grabbing information from the quadrature
  auto x_interpolator = x.FESpace()->GetQuadratureInterpolator(zone_dofs);
  x_interpolator->SetOutputLayout(QVectorLayout::byNODES);
  Vector u_HO_dofs(dim * num_ldofs * NE);

  // Evaluates the right hand side gridfunction at x
  x_interpolator->Values(x_local, u_HO_dofs);

  // temporary vector for the interpolating
  Vector HO_dofs(u_HO_dofs);

  auto u_LO_view = mfem::Reshape(u_HO_dofs.Read(), num_ldofs, dim, NE);
  auto u_LO_xyz_view = mfem::Reshape(HO_dofs.Write(), num_ldofs * NE, dim);

  for (int e = 0; e < NE; ++e) {
    for (int i = 0; i < num_ldofs; ++i) {
      int ti = i + num_ldofs * e;
      u_LO_xyz_view(ti, 0) = u_LO_view(i, 0, e);
      u_LO_xyz_view(ti, 1) = u_LO_view(i, 1, e);
    }
  }

  finder.Interpolate(HO_dofs, u_LOR, u_HO_interp);
}

// Calculate the Low Order solution
void CalculateLOSolution(const GridFunction &u_HO, const GridFunction &x,
                         const double &dt, GridFunction &u_LO, Vector &el_mass,
                         Vector &el_vol) {
  // Grabbing information from the finite element space
  auto *Tr = x.FESpace()->GetMesh()->GetElementTransformation(0);
  const int NE = x.FESpace()->GetNE();

  const FiniteElement *fe = u_HO.FESpace()->GetFE(0);
  const IntegrationRule &ir = MassIntegrator::GetRule(*fe, *fe, *Tr);
  const int nqp = ir.GetNPoints();

  // Grabbing information from the quadrature
  GeometricFactors geom(x, ir, GeometricFactors::DETERMINANTS);
  auto qi_u = u_HO.FESpace()->GetQuadratureInterpolator(ir);
  Vector u_qvals(nqp * NE);
  qi_u->Values(u_HO, u_qvals);

  // Quadrature for calculating the mass and volume
  for (int k = 0; k < NE; k++) {
    el_mass(k) = 0.0;
    el_vol(k) = 0.0;
    for (int q = 0; q < nqp; q++) {
      const IntegrationPoint &ip = ir.IntPoint(q);
      el_mass(k) += ip.weight * geom.detJ(k * nqp + q) * u_qvals(k * nqp + q);
      el_vol(k) += ip.weight * geom.detJ(k * nqp + q);
    }
  }

  // Apply the averaging
  const int ndofs = u_HO.Size() / NE;
  for (int k = 0; k < NE; k++) {
    double zone_avg = el_mass(k) / el_vol(k);
    for (int i = 0; i < ndofs; i++) {
      u_LO(k * ndofs + i) = zone_avg;
    }
  }
}

// Updating the time step based on the updated values for u, pretty much
// take diectly from the Remhos repo
void UpdateTimeStepEstimate(const Vector &x, const Vector &dx,
                            const Vector &x_min, const Vector &x_max,
                            double &dt_est) {
  // if (dt_control == TimeStepControl::FixedTimeStep) { return; }

  // x_min <= x + dt * dx <= x_max.
  int n = x.Size();
  int NE = x_min.Size();
  const int ndofs = n / NE;
  // cout << "n = " << n << endl;
  // cout << "NE  = " << NE << endl;
  // cout << "size of u_min is = " << x_min.Size() << endl;
  // cout << "size of u_max is = " << x_max.Size() << endl;
  const double eps = 1e-12;
  double dt = numeric_limits<double>::infinity();

  // I have a fundamental misunderstanding on this part
  for (int i = 0; i < NE; i++) {
    if (dx(i * ndofs) > eps) {
      dt = fmin(dt, (x_max(i) - x(i * ndofs)) / dx(i * ndofs));
    } else if (dx(i * ndofs) < -eps) {
      dt = fmin(dt, (x_min(i) - x(i * ndofs)) / dx(i * ndofs));
    }
  }

  // MPI_Allreduce(MPI_IN_PLACE, &dt, 1, MPI_DOUBLE, MPI_MIN,
  //              Kbf.ParFESpace()->GetComm());

  dt_est = fmin(dt_est, dt);
}

// Calculate the min and max of an element
// This function is essentially the same as the one found in Remhos,
// which is titled the same.
void ComputeElementsMinMax(const Vector &u, const FiniteElementSpace &fes,
                           Vector &u_min, Vector &u_max, Array<bool> *active_el,
                           Array<bool> *active_dof) {
  const int NE = fes.GetNE(), ndof = fes.GetFE(0)->GetDof();
  int dof_id;
  u.HostRead();
  u_min.HostReadWrite();
  u_max.HostReadWrite();
  for (int k = 0; k < NE; k++) {
    u_min(k) = numeric_limits<double>::infinity();
    u_max(k) = -numeric_limits<double>::infinity();

    // Inactive elements don't affect the bounds
    if (active_el && (*active_el)[k] == false) {
      continue;
    }

    for (int i = 0; i < ndof; i++) {
      dof_id = k * ndof + i;
      // Inactive dofs don't affect the bounds.
      if (active_dof && (*active_dof)[dof_id] == false) {
        continue;
      }

      u_min(k) = min(u_min(k), u(dof_id));
      u_max(k) = max(u_max(k), u(dof_id));
    }
  }
}

// Visualization function
void visualize(VisItDataCollection &dc, string prefix, int x, int y) {
  int w = Ww, h = Wh;

  char vishost[] = "localhost";
  int visport = 19916;

  socketstream sol_sockL2(vishost, visport);
  sol_sockL2.precision(8);
  sol_sockL2 << "solution\n"
             << *dc.GetMesh() << *dc.GetField("Density") << "window_geometry "
             << x << " " << y << " " << w << " " << h << "plot_caption '"
             << " " << prefix << " Density'"
             << "'" << flush;
}

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(BilinearForm &M_, BilinearForm &K_, const Vector &b_)
    : TimeDependentOperator(M_.Height()), M(M_), K(K_), b(b_), z(M_.Height()) {
  Array<int> ess_tdof_list;
  if (M.GetAssemblyLevel() == AssemblyLevel::LEGACY) {
    M_prec = new DSmoother(M.SpMat());
    M_solver.SetOperator(M.SpMat());
    dg_solver = new DG_Solver(M.SpMat(), K.SpMat(), *M.FESpace());
  } else {
    M_prec = new OperatorJacobiSmoother(M, ess_tdof_list);
    M_solver.SetOperator(M);
    dg_solver = NULL;
  }
  M_solver.SetPreconditioner(*M_prec);
  M_solver.iterative_mode = false;
  M_solver.SetRelTol(1e-9);
  M_solver.SetAbsTol(0.0);
  M_solver.SetMaxIter(100);
  M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const {
  // y = M^{-1} (K x + b)
  K.Mult(x, z);
  z += b;
  M_solver.Mult(z, y);
}

void FE_Evolution::ImplicitSolve(const double dt, const Vector &x, Vector &k) {
  MFEM_VERIFY(
      dg_solver != NULL,
      "Implicit time integration is not supported with partial assembly");
  K.Mult(x, z);
  z += b;
  dg_solver->SetTimeStep(dt);
  dg_solver->Mult(z, k);
}

FE_Evolution::~FE_Evolution() {
  delete M_prec;
  delete dg_solver;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v) {
  int dim = x.Size();

  // map to the reference [-1,1] domain
  Vector X(dim);
  for (int i = 0; i < dim; i++) {
    double center = (bb_min[i] + bb_max[i]) * 0.5;
    X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
  }

  switch (problem) {
  case 0: {
    // Translations in 1D, 2D, and 3D
    switch (dim) {
    case 1:
      v(0) = 1.0;
      break;
    case 2:
      v(0) = sqrt(2. / 3.);
      v(1) = sqrt(1. / 3.);
      break;
    case 3:
      v(0) = sqrt(3. / 6.);
      v(1) = sqrt(2. / 6.);
      v(2) = sqrt(1. / 6.);
      break;
    }
    break;
  }
  case 1:
  case 2: {
    // Clockwise rotation in 2D around the origin
    const double w = M_PI / 2;
    switch (dim) {
    case 1:
      v(0) = 1.0;
      break;
    case 2:
      v(0) = w * X(1);
      v(1) = -w * X(0);
      break;
    case 3:
      v(0) = w * X(1);
      v(1) = -w * X(0);
      v(2) = 0.0;
      break;
    }
    break;
  }
  case 3: {
    // Clockwise twisting rotation in 2D around the origin
    const double w = M_PI / 2;
    double d =
        max((X(0) + 1.) * (1. - X(0)), 0.) * max((X(1) + 1.) * (1. - X(1)), 0.);
    d = d * d;
    switch (dim) {
    case 1:
      v(0) = 1.0;
      break;
    case 2:
      v(0) = d * w * X(1);
      v(1) = -d * w * X(0);
      break;
    case 3:
      v(0) = d * w * X(1);
      v(1) = -d * w * X(0);
      v(2) = 0.0;
      break;
    }
    break;
  }
  }
}

// Initial condition
double u0_function(const Vector &x) {
  int dim = x.Size();

  // map to the reference [-1,1] domain
  Vector X(dim);
  for (int i = 0; i < dim; i++) {
    double center = (bb_min[i] + bb_max[i]) * 0.5;
    X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
  }

  switch (problem) {
  case 0:
  case 1: {
    switch (dim) {
    case 1:
      return exp(-40. * pow(X(0) - 0.5, 2));
    case 2:
    case 3: {
      double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
      if (dim == 3) {
        const double s = (1. + 0.25 * cos(2 * M_PI * X(2)));
        rx *= s;
        ry *= s;
      }
      return (erfc(w * (X(0) - cx - rx)) * erfc(-w * (X(0) - cx + rx)) *
              erfc(w * (X(1) - cy - ry)) * erfc(-w * (X(1) - cy + ry))) /
             16;
    }
    }
  }
  case 2: {
    double x_ = X(0), y_ = X(1), rho, phi;
    rho = hypot(x_, y_);
    phi = atan2(y_, x_);
    return pow(sin(M_PI * rho), 2) * sin(3 * phi);
  }
  case 3: {
    const double f = M_PI;
    return sin(f * X(0)) * sin(f * X(1));
  }
  }
  return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x) {
  switch (problem) {
  case 0:
  case 1:
  case 2:
  case 3:
    return 0.0;
  }
  return 0.0;
}

void FCT_Project(DenseMatrix &M, DenseMatrixInverse &M_inv, Vector &m,
                 Vector &x, double y_min, double y_max, Vector &xy) {
  // [IN]  - M, M_inv, m, x, y_min, y_max
  // [OUT] - xy

  m.HostReadWrite();
  x.HostReadWrite();
  xy.HostReadWrite();
  const int s = M.Size();

  xy.SetSize(s);

  // Compute the lumped mass matrix in ML
  Vector ML(s);
  M.GetRowSums(ML);

  // Compute the high-order projection in xy
  M_inv.Mult(m, xy);

  // Q0 solutions can't be adjusted conservatively. It's what it is.
  if (xy.Size() == 1) {
    return;
  }

  double dMLX(0);
  for (int i = 0; i < x.Size(); ++i) {
    dMLX += ML(i) * x(i);
  }

  const double y_avg = m.Sum() / dMLX;

  if (!(y_min < y_avg + 1e-12 && y_avg < y_max + 1e-12)) {
    std::cout << "Average is out of bounds: "
              << "y_min < y_avg + 1e-12 && y_avg < y_max + 1e-12 " << y_min
              << " " << y_avg << " " << y_max << std::endl;
  }

  Vector z(s);
  Vector beta(s);
  Vector Mxy(s);
  M.Mult(xy, Mxy);
  for (int i = 0; i < s; i++) {
    // Some different options for beta:
    // beta(i) = 1.0;
    beta(i) = ML(i) * x(i);
    // beta(i) = ML(i)*(x(i) + 1e-14);
    // beta(i) = ML(i);
    // beta(i) = Mxy(i);

    // The low order flux correction
    z(i) = m(i) - ML(i) * x(i) * y_avg;
  }

  // Make beta_i sum to 1
  beta /= beta.Sum();

  DenseMatrix F(s);
  for (int i = 1; i < s; i++) {
    for (int j = 0; j < i; j++) {
      F(i, j) = M(i, j) * (xy(i) - xy(j)) + (beta(j) * z(i) - beta(i) * z(j));
    }
  }

  Vector gp(s), gm(s);
  gp = 0.0;
  gm = 0.0;
  for (int i = 1; i < s; i++) {
    for (int j = 0; j < i; j++) {
      double fij = F(i, j);
      if (fij >= 0.0) {
        gp(i) += fij;
        gm(j) -= fij;
      } else {
        gm(i) += fij;
        gp(j) -= fij;
      }
    }
  }

  for (int i = 0; i < s; i++) {
    xy(i) = x(i) * y_avg;
  }

  for (int i = 0; i < s; i++) {
    double mi = ML(i), xyLi = xy(i);
    double rp = std::max(mi * (x(i) * y_max - xyLi), 0.0);
    double rm = std::min(mi * (x(i) * y_min - xyLi), 0.0);
    double sp = gp(i), sm = gm(i);

    gp(i) = (rp < sp) ? rp / sp : 1.0;
    gm(i) = (rm > sm) ? rm / sm : 1.0;
  }

  for (int i = 1; i < s; i++) {
    for (int j = 0; j < i; j++) {
      double fij = F(i, j), aij;

      if (fij >= 0.0) {
        aij = std::min(gp(i), gm(j));
      } else {
        aij = std::min(gm(i), gp(j));
      }

      fij *= aij;
      xy(i) += fij / ML(i);
      xy(j) -= fij / ML(j);
    }
  }
}