# Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# Sets the following variables:
#   - Scotch_FOUND
#   - Scotch_INCLUDE_DIRS
#   - Scotch_LIBRARIES

include(MfemCmakeUtilities)
mfem_find_package(Scotch Scotch Scotch_DIR "" "" "" ""
  "Paths to headers required by Scotch."
  "Libraries required by Scotch."
  ADD_COMPONENT "scotch" "include" scotch.h "lib" scotch
  ADD_COMPONENT "scotcherr" "" "" "lib" scotcherr
  ADD_COMPONENT "scotcherrexit" "" "" "lib" scotcherrexit
  ADD_COMPONENT "scotchmetis" "include" "metis.h" "lib" scotchmetis
  ADD_COMPONENT "scotchmetisv3" "include" "metis.h" "lib" scotchmetisv3
  ADD_COMPONENT "scotchmetisv5" "include" "metis.h" "lib" scotchmetisv5
  ADD_COMPONENT "esmumps" "" "" "lib" esmumps
  ADD_COMPONENT "ptscotch" "include" ptscotch.h "lib" ptscotch
  ADD_COMPONENT "ptscotcherr" "" "" "lib" ptscotcherr
  ADD_COMPONENT "ptscotcherrexit" "" "" "lib" ptscotcherrexit
  ADD_COMPONENT "ptscotchparmetis" "include" "parmetis.h" "lib" ptscotchparmetis
  ADD_COMPONENT "ptscotchparmetisv3" "include" "parmetis.h" "lib" ptscotchparmetisv3
  ADD_COMPONENT "ptesmumps" "" "" "lib" ptesmumps
  )
