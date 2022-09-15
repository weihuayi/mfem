// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"
#include "../fem/kernels.hpp"
#include "../general/forall.hpp"

namespace mfem
{

template<int T_D1D = 0, int T_Q1D = 0> static
void BLFEvalAssemble2D(const int vdim, const int nbe, const int d, const int q,
                       const bool normals, const int *markers, const double *b,
                       const double *detj, const double *n, const double *weights,
                       const Vector &coeff, double *y)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, nbe);
   const auto B = Reshape(b, q, d);
   const auto detJ = Reshape(detj, q, nbe);
   const auto N = Reshape(n, q, 2, nbe);
   const auto W = Reshape(weights, q);
   const int cvdim = normals ? 2 : 1;
   const bool cst = coeff.Size() == cvdim;
   const auto C = cst ? Reshape(F,cvdim,1,1) : Reshape(F,cvdim,q,nbe);
   auto Y = Reshape(y, d, vdim, nbe);

   MFEM_FORALL(e, nbe,
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : MAX_Q1D;
      double QQ[Q];

      for (int c = 0; c < vdim; ++c)
      {
         for (int qx = 0; qx < q; ++qx)
         {
            double coeff_val = 0.0;
            if (normals)
            {
               for (int cd = 0; cd < 2; ++cd)
               {
                  const double cval = cst ? C(cd,0,0) : C(cd,qx,e);
                  coeff_val += cval * N(qx, cd, e);
               }
            }
            else
            {
               coeff_val = cst ? C(0,0,0) : C(0,qx,e);
            }
            QQ[qx] = W(qx) * coeff_val * detJ(qx,e);
         }
         for (int dx = 0; dx < d; ++dx)
         {
            double u = 0;
            for (int qx = 0; qx < q; ++qx) { u += QQ[qx] * B(qx,dx); }
            Y(dx,c,e) += u;
         }
      }
   });
}

template<int T_D1D = 0, int T_Q1D = 0> static
void BLFEvalAssemble3D(const int vdim, const int nbe, const int d, const int q,
                       const bool normals, const int *markers, const double *b,
                       const double *detj, const double *n, const double *weights,
                       const Vector &coeff, double *y)
{
   const auto F = coeff.Read();
   const auto M = Reshape(markers, nbe);
   const auto B = Reshape(b, q, d);
   const auto detJ = Reshape(detj, q, q, nbe);
   const auto N = Reshape(n, q, q, 3, nbe);
   const auto W = Reshape(weights, q, q);
   const int cvdim = normals ? 3 : 1;
   const bool cst = coeff.Size() == cvdim;
   const auto C = cst ? Reshape(F,cvdim,1,1,1) : Reshape(F,cvdim,q,q,nbe);
   auto Y = Reshape(y, d, d, vdim, nbe);

   MFEM_FORALL_2D(e, nbe, q, q, 1,
   {
      if (M(e) == 0) { return; } // ignore

      constexpr int Q = T_Q1D ? T_Q1D : MAX_Q1D;
      constexpr int D = T_D1D ? T_D1D : MAX_D1D;

      MFEM_SHARED double sBt[Q*D];
      MFEM_SHARED double sQQ[Q*Q];
      MFEM_SHARED double sQD[Q*D];

      const DeviceMatrix Bt(sBt, d, q);
      kernels::internal::LoadB<D,Q>(d, q, B, sBt);

      const DeviceMatrix QQ(sQQ, q, q);
      const DeviceMatrix QD(sQD, q, d);

      for (int c = 0; c < vdim; ++c)
      {
         MFEM_FOREACH_THREAD(x,x,q)
         {
            MFEM_FOREACH_THREAD(y,y,q)
            {
               double coeff_val = 0.0;
               if (normals)
               {
                  for (int cd = 0; cd < 3; ++cd)
                  {
                     double cval = cst ? C(cd,0,0,0) : C(cd,x,y,e);
                     coeff_val += cval * N(x,y,cd,e);
                  }
               }
               else
               {
                  coeff_val = cst ? C(0,0,0,0) : C(0,x,y,e);
               }
               QQ(y,x) = W(x,y) * coeff_val * detJ(x,y,e);
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(qy,y,q)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               double u = 0.0;
               for (int qx = 0; qx < q; ++qx) { u += QQ(qy,qx) * Bt(dx,qx); }
               QD(qy,dx) = u;
            }
         }
         MFEM_SYNC_THREAD;
         MFEM_FOREACH_THREAD(dy,y,d)
         {
            MFEM_FOREACH_THREAD(dx,x,d)
            {
               double u = 0.0;
               for (int qy = 0; qy < q; ++qy) { u += QD(qy,dx) * Bt(dy,qy); }
               Y(dx,dy,c,e) += u;
            }
         }
         MFEM_SYNC_THREAD;
      }
   });
}

static void BLFEvalAssemble(const FiniteElementSpace &fes,
                            const IntegrationRule &ir,
                            const Array<int> &markers,
                            const Vector &coeff,
                            const bool normals,
                            Vector &y)
{
   Mesh &mesh = *fes.GetMesh();
   const int dim = mesh.Dimension();
   const FiniteElement &el = *fes.GetBE(0);
   const MemoryType mt = Device::GetDeviceMemoryType();
   const DofToQuad &maps = el.GetDofToQuad(ir, DofToQuad::TENSOR);
   const int d = maps.ndof, q = maps.nqpt;
   int flags = FaceGeometricFactors::DETERMINANTS;
   if (normals) { flags |= FaceGeometricFactors::NORMALS; }
   const FaceGeometricFactors *geom = mesh.GetFaceGeometricFactors(
                                         ir, flags, FaceType::Boundary, mt);
   auto ker = (dim == 2) ? BLFEvalAssemble2D<> : BLFEvalAssemble3D<>;

   if (dim==2)
   {
      if (d==1 && q==1) { ker=BLFEvalAssemble2D<1,1>; }
      if (d==2 && q==2) { ker=BLFEvalAssemble2D<2,2>; }
      if (d==3 && q==3) { ker=BLFEvalAssemble2D<3,3>; }
      if (d==4 && q==4) { ker=BLFEvalAssemble2D<4,4>; }
      if (d==5 && q==5) { ker=BLFEvalAssemble2D<5,5>; }
      if (d==2 && q==3) { ker=BLFEvalAssemble2D<2,3>; }
      if (d==3 && q==4) { ker=BLFEvalAssemble2D<3,4>; }
      if (d==4 && q==5) { ker=BLFEvalAssemble2D<4,5>; }
      if (d==5 && q==6) { ker=BLFEvalAssemble2D<5,6>; }
   }

   if (dim==3)
   {
      if (d==1 && q==1) { ker=BLFEvalAssemble3D<1,1>; }
      if (d==2 && q==2) { ker=BLFEvalAssemble3D<2,2>; }
      if (d==3 && q==3) { ker=BLFEvalAssemble3D<3,3>; }
      if (d==4 && q==4) { ker=BLFEvalAssemble3D<4,4>; }
      if (d==5 && q==5) { ker=BLFEvalAssemble3D<5,5>; }
      if (d==2 && q==3) { ker=BLFEvalAssemble3D<2,3>; }
      if (d==3 && q==4) { ker=BLFEvalAssemble3D<3,4>; }
      if (d==4 && q==5) { ker=BLFEvalAssemble3D<4,5>; }
      if (d==5 && q==6) { ker=BLFEvalAssemble3D<5,6>; }
   }

   MFEM_VERIFY(ker, "No kernel ndof " << d << " nqpt " << q);

   const int vdim = fes.GetVDim();
   const int nbe = fes.GetMesh()->GetNFbyType(FaceType::Boundary);
   const int *M = markers.Read();
   const double *B = maps.B.Read();
   const double *detJ = geom->detJ.Read();
   const double *n = geom->normal.Read();
   const double *W = ir.GetWeights().Read();
   double *Y = y.ReadWrite();
   ker(vdim, nbe, d, q, normals, M, B, detJ, n, W, coeff, Y);
}

void BoundaryLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                          const Array<int> &markers,
                                          Vector &b)
{
   const FiniteElement &fe = *fes.GetBE(0);
   const int qorder = oa * fe.GetOrder() + ob;
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule &ir = IntRule ? *IntRule : IntRules.Get(gtype, qorder);
   const int nq = ir.GetNPoints();
   Mesh &mesh = *fes.GetMesh();
   const int nbe = mesh.GetNFbyType(FaceType::Boundary);
   const int dim = mesh.Dimension();
   const DofToQuad &maps = fe.GetDofToQuad(ir, DofToQuad::TENSOR);

   Vector coeff;
   if (ConstantCoefficient *cQ =
          dynamic_cast<ConstantCoefficient*>(&Q))
   {
      coeff.SetSize(1);
      coeff(0) = cQ->constant;
   }
   else
   {
      coeff.SetSize(nq * nbe);
      auto C = Reshape(coeff.HostWrite(), nq, nbe);
      int f_ind = 0;
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const Mesh::FaceInformation face = mesh.GetFaceInformation(f);
         if (!face.IsBoundary()) { continue; }
         ElementTransformation &Tr = *mesh.GetFaceElementTransformations(f);
         for (int q = 0; q < nq; ++q)
         {
            int iq = ToLexOrdering(dim, face.element[0].local_face_id,
                                   maps.nqpt, q);
            const IntegrationPoint &ip = ir[iq];
            Tr.SetIntPoint(&ip);
            C(q,f_ind) = Q.Eval(Tr, ip);
         }
         f_ind++;
      }
   }
   BLFEvalAssemble(fes, ir, markers, coeff, false, b);
}

void BoundaryNormalLFIntegrator::AssembleDevice(const FiniteElementSpace &fes,
                                                const Array<int> &markers,
                                                Vector &b)
{
   const FiniteElement &fe = *fes.GetBE(0);
   const int qorder = oa * fe.GetOrder() + ob;
   const Geometry::Type gtype = fe.GetGeomType();
   const IntegrationRule &ir = IntRule ? *IntRule : IntRules.Get(gtype, qorder);
   const int nq = ir.GetNPoints();
   Mesh &mesh = *fes.GetMesh();
   const int nbe = mesh.GetNFbyType(FaceType::Boundary);
   const int dim = mesh.Dimension();
   const DofToQuad &maps = fe.GetDofToQuad(ir, DofToQuad::TENSOR);

   Vector coeff;
   if (const auto *cQ = dynamic_cast<VectorConstantCoefficient*>(&Q))
   {
      coeff = cQ->GetVec();
   }
   else
   {

      Vector coeff_val(dim);
      coeff.SetSize(dim * nq * nbe);
      auto C = Reshape(coeff.HostWrite(), dim, nq, nbe);
      int f_ind = 0;
      for (int f = 0; f < fes.GetNF(); ++f)
      {
         const Mesh::FaceInformation face = mesh.GetFaceInformation(f);
         if (!face.IsBoundary()) { continue; }
         ElementTransformation &Tr = *mesh.GetFaceElementTransformations(f);
         for (int q = 0; q < nq; ++q)
         {
            int iq = ToLexOrdering(dim, face.element[0].local_face_id,
                                   maps.nqpt, q);
            const IntegrationPoint &ip = ir[iq];
            Tr.SetIntPoint(&ip);
            Q.Eval(coeff_val, Tr, ip);
            for (int d=0; d<dim; ++d) { C(d,q,f_ind) = coeff_val[d]; }
         }
         f_ind++;
      }
   }
   BLFEvalAssemble(fes, ir, markers, coeff, true, b);
}

} // namespace mfem
