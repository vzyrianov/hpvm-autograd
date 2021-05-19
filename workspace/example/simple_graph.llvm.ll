; ModuleID = 'simple_graph.hpvm.ll'
source_filename = "simple_graph.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl" }
%"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl" = type { %"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl_data" }
%"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl_data" = type { float*, float*, float* }
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", i64, %union.anon }
%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%union.anon = type { i64, [8 x i8] }
%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_get"* }
%"class.std::ios_base" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"class.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"class.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i64 }
%"class.std::locale" = type { %"class.std::locale::_Impl"* }
%"class.std::locale::_Impl" = type { i32, %"class.std::locale::facet"**, i64, %"class.std::locale::facet"**, i8** }
%"class.std::locale::facet" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_streambuf" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale" }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque
%"class.std::num_put" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::num_get" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::allocator" = type { i8 }
%"class.std::allocator.0" = type { i8 }
%"class.__gnu_cxx::new_allocator.1" = type { i8 }
%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque
%struct.Tensor = type { i32, i32, i32, i32, %struct.cudnnTensorStruct*, %struct.cudnnFilterStruct*, %struct.cudnnTensorStruct*, %struct.cudnnFilterStruct*, i8*, i8*, i8*, i64, i64, %struct.Dimension }
%struct.cudnnTensorStruct = type opaque
%struct.cudnnFilterStruct = type opaque
%struct.Dimension = type { i32, i64* }
%"class.std::__cxx11::basic_ostringstream" = type { %"class.std::basic_ostream.base", %"class.std::__cxx11::basic_stringbuf", %"class.std::basic_ios" }
%"class.std::basic_ostream.base" = type { i32 (...)** }
%"class.std::__cxx11::basic_stringbuf" = type { %"class.std::basic_streambuf", i32, %"class.std::__cxx11::basic_string" }
%"class.std::vector.3" = type { %"struct.std::_Vector_base.4" }
%"struct.std::_Vector_base.4" = type { %"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl" }
%"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl" = type { %"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl_data" }
%"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl_data" = type { %struct.ClassProb*, %struct.ClassProb*, %struct.ClassProb* }
%struct.ClassProb = type { float, i32 }
%"class.std::allocator.5" = type { i8 }
%"class.__gnu_cxx::__normal_iterator" = type { %struct.ClassProb* }
%"class.__gnu_cxx::new_allocator" = type { i8 }
%"class.__gnu_cxx::new_allocator.6" = type { i8 }
%"class.__gnu_cxx::__normal_iterator.8" = type { i8* }
%"class.__gnu_cxx::__normal_iterator.9" = type { i8* }
%"struct.__gnu_cxx::__ops::_Iter_comp_iter" = type { i1 (i64, i64)* }
%"struct.__gnu_cxx::__ops::_Iter_comp_val" = type { i1 (i64, i64)* }
%"struct.__gnu_cxx::__ops::_Val_comp_iter" = type { i1 (i64, i64)* }
%"class.__gnu_cxx::__normal_iterator.10" = type { float* }
%struct.out._Z10var_0_nodePvmS_m = type <{ i8*, i64 }>
%struct.out._Z10var_1_nodePvm = type <{ i8*, i64 }>
%struct.out._Z10var_2_nodePvm = type <{ i8*, i64 }>
%struct.out._Z4rootPvmS_m = type <{ i8*, i64 }>
%_Z4rootPvmS_m_cloned.4.arg.struct.ty = type <{ i8*, i64, i8*, i64, %struct.out._Z4rootPvmS_m }>

$_ZNSt6vectorIfSaIfEEC2Ev = comdat any

$_ZNSt6vectorIfSaIfEED2Ev = comdat any

$_ZSt5fixedRSt8ios_base = comdat any

$_ZNSt6vectorI9ClassProbSaIS0_EEC2Ev = comdat any

$_ZNSt6vectorI9ClassProbSaIS0_EE9push_backERKS0_ = comdat any

$_ZSt4sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEEPFbS2_S2_EEvT_SA_T0_ = comdat any

$_ZNSt6vectorI9ClassProbSaIS0_EE5beginEv = comdat any

$_ZNSt6vectorI9ClassProbSaIS0_EE3endEv = comdat any

$_ZNSt6vectorI9ClassProbSaIS0_EEixEm = comdat any

$_ZNSt6vectorI9ClassProbSaIS0_EED2Ev = comdat any

$_ZNSt6vectorIfSaIfEE9push_backERKf = comdat any

$_ZNKSt6vectorIfSaIfEE4sizeEv = comdat any

$_ZNSt6vectorIfSaIfEEixEm = comdat any

$_ZSt5log10f = comdat any

$_ZSt4sqrtf = comdat any

$_ZNSt12_Vector_baseIfSaIfEEC2Ev = comdat any

$_ZNSt12_Vector_baseIfSaIfEE12_Vector_implC2Ev = comdat any

$_ZNSaIfEC2Ev = comdat any

$_ZNSt12_Vector_baseIfSaIfEE17_Vector_impl_dataC2Ev = comdat any

$_ZN9__gnu_cxx13new_allocatorIfEC2Ev = comdat any

$_ZNSt8ios_base4setfESt13_Ios_FmtflagsS0_ = comdat any

$_ZStaNRSt13_Ios_FmtflagsS_ = comdat any

$_ZStcoSt13_Ios_Fmtflags = comdat any

$_ZStoRRSt13_Ios_FmtflagsS_ = comdat any

$_ZStanSt13_Ios_FmtflagsS_ = comdat any

$_ZStorSt13_Ios_FmtflagsS_ = comdat any

$_ZNSt12_Vector_baseI9ClassProbSaIS0_EEC2Ev = comdat any

$_ZNSt12_Vector_baseI9ClassProbSaIS0_EE12_Vector_implC2Ev = comdat any

$_ZNSaI9ClassProbEC2Ev = comdat any

$_ZNSt12_Vector_baseI9ClassProbSaIS0_EE17_Vector_impl_dataC2Ev = comdat any

$_ZN9__gnu_cxx13new_allocatorI9ClassProbEC2Ev = comdat any

$__clang_call_terminate = comdat any

$_ZN9__gnu_cxx13new_allocatorIcEC2Ev = comdat any

$_ZN9__gnu_cxx13new_allocatorIcED2Ev = comdat any

$_ZNSt14pointer_traitsIPKcE10pointer_toERS0_ = comdat any

$_ZSt9addressofIKcEPT_RS1_ = comdat any

$_ZSt11__addressofIKcEPT_RS1_ = comdat any

$_ZNSt16allocator_traitsISaIcEE10deallocateERS0_Pcm = comdat any

$_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm = comdat any

$_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_ = comdat any

$_ZNSt11char_traitsIcE6lengthEPKc = comdat any

$_ZNSt14pointer_traitsIPcE10pointer_toERc = comdat any

$_ZSt9addressofIcEPT_RS0_ = comdat any

$_ZSt11__addressofIcEPT_RS0_ = comdat any

$_ZN9__gnu_cxx13new_allocatorIcEC2ERKS1_ = comdat any

$_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_construct_auxIPKcEEvT_S8_St12__false_type = comdat any

$_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag = comdat any

$_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_ = comdat any

$_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_ = comdat any

$_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag = comdat any

$_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_ = comdat any

$_ZNSt11char_traitsIcE6assignERcRKc = comdat any

$_ZNSt11char_traitsIcE4copyEPcPKcm = comdat any

$_ZSt8_DestroyIPffEvT_S1_RSaIT0_E = comdat any

$_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv = comdat any

$_ZNSt12_Vector_baseIfSaIfEED2Ev = comdat any

$_ZSt8_DestroyIPfEvT_S1_ = comdat any

$_ZNSt12_Destroy_auxILb1EE9__destroyIPfEEvT_S3_ = comdat any

$_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm = comdat any

$_ZN9__gnu_cxx13new_allocatorIfED2Ev = comdat any

$_ZNSt16allocator_traitsISaIfEE10deallocateERS0_Pfm = comdat any

$_ZN9__gnu_cxx13new_allocatorIfE10deallocateEPfm = comdat any

$_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEED2Ev = comdat any

$_ZSt4moveIRSaIcEEONSt16remove_referenceIT_E4typeEOS3_ = comdat any

$_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignIPcvEERS4_T_S8_ = comdat any

$_ZN9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEC2IPcEERKNS0_IT_NS_11__enable_ifIXsr3std10__are_sameISC_SB_EE7__valueES8_E6__typeEEE = comdat any

$_ZN9__gnu_cxxmiIPKcPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEDTmicldtfp_4baseEcldtfp0_4baseEERKNS_17__normal_iteratorIT_T1_EERKNSB_IT0_SD_EE = comdat any

$_ZN9__gnu_cxxmiIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKSC_SF_ = comdat any

$_ZNK9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE4baseEv = comdat any

$_ZNK9__gnu_cxx17__normal_iteratorIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE4baseEv = comdat any

$_ZN9__gnu_cxx17__normal_iteratorIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEC2ERKS1_ = comdat any

$_ZSt8_DestroyIP9ClassProbS0_EvT_S2_RSaIT0_E = comdat any

$_ZNSt12_Vector_baseI9ClassProbSaIS0_EE19_M_get_Tp_allocatorEv = comdat any

$_ZNSt12_Vector_baseI9ClassProbSaIS0_EED2Ev = comdat any

$_ZSt8_DestroyIP9ClassProbEvT_S2_ = comdat any

$_ZNSt12_Destroy_auxILb1EE9__destroyIP9ClassProbEEvT_S4_ = comdat any

$_ZNSt12_Vector_baseI9ClassProbSaIS0_EE13_M_deallocateEPS0_m = comdat any

$_ZN9__gnu_cxx13new_allocatorI9ClassProbED2Ev = comdat any

$_ZNSt16allocator_traitsISaI9ClassProbEE10deallocateERS1_PS0_m = comdat any

$_ZN9__gnu_cxx13new_allocatorI9ClassProbE10deallocateEPS1_m = comdat any

$_ZNSt16allocator_traitsISaI9ClassProbEE9constructIS0_JRKS0_EEEvRS1_PT_DpOT0_ = comdat any

$_ZNSt6vectorI9ClassProbSaIS0_EE17_M_realloc_insertIJRKS0_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_ = comdat any

$_ZN9__gnu_cxx13new_allocatorI9ClassProbE9constructIS1_JRKS1_EEEvPT_DpOT0_ = comdat any

$_ZSt7forwardIRK9ClassProbEOT_RNSt16remove_referenceIS3_E4typeE = comdat any

$_ZNKSt6vectorI9ClassProbSaIS0_EE12_M_check_lenEmPKc = comdat any

$_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_ = comdat any

$_ZNSt12_Vector_baseI9ClassProbSaIS0_EE11_M_allocateEm = comdat any

$_ZNSt6vectorI9ClassProbSaIS0_EE11_S_relocateEPS0_S3_S3_RS1_ = comdat any

$_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv = comdat any

$_ZNKSt6vectorI9ClassProbSaIS0_EE8max_sizeEv = comdat any

$_ZNKSt6vectorI9ClassProbSaIS0_EE4sizeEv = comdat any

$_ZSt3maxImERKT_S2_S2_ = comdat any

$_ZNSt6vectorI9ClassProbSaIS0_EE11_S_max_sizeERKS1_ = comdat any

$_ZNKSt12_Vector_baseI9ClassProbSaIS0_EE19_M_get_Tp_allocatorEv = comdat any

$_ZNSt16allocator_traitsISaI9ClassProbEE8max_sizeERKS1_ = comdat any

$_ZSt3minImERKT_S2_S2_ = comdat any

$_ZNK9__gnu_cxx13new_allocatorI9ClassProbE8max_sizeEv = comdat any

$_ZNSt16allocator_traitsISaI9ClassProbEE8allocateERS1_m = comdat any

$_ZN9__gnu_cxx13new_allocatorI9ClassProbE8allocateEmPKv = comdat any

$_ZNSt6vectorI9ClassProbSaIS0_EE14_S_do_relocateEPS0_S3_S3_RS1_St17integral_constantIbLb1EE = comdat any

$_ZSt12__relocate_aIP9ClassProbS1_SaIS0_EET0_T_S4_S3_RT1_ = comdat any

$_ZSt14__relocate_a_1I9ClassProbS0_ENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS2_E4typeES3_S3_S3_RSaIT0_E = comdat any

$_ZSt12__niter_baseIP9ClassProbET_S2_ = comdat any

$_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEC2ERKS2_ = comdat any

$_ZSt6__sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_ = comdat any

$_ZN9__gnu_cxx5__ops16__iter_comp_iterIPFb9ClassProbS2_EEENS0_15_Iter_comp_iterIT_EES6_ = comdat any

$_ZN9__gnu_cxxneIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_ = comdat any

$_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElNS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_T1_ = comdat any

$_ZSt4__lgl = comdat any

$_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_ = comdat any

$_ZSt14__partial_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_T0_ = comdat any

$_ZSt27__unguarded_partition_pivotIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEET_SD_SD_T0_ = comdat any

$_ZSt13__heap_selectIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_T0_ = comdat any

$_ZSt11__sort_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_RT0_ = comdat any

$_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_RT0_ = comdat any

$_ZN9__gnu_cxxltIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_ = comdat any

$_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_ = comdat any

$_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_RT0_ = comdat any

$_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEppEv = comdat any

$_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_ = comdat any

$_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl = comdat any

$_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv = comdat any

$_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElS2_NS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_T0_SE_T1_T2_ = comdat any

$_ZSt4moveIRN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS3_EEEEONSt16remove_referenceIT_E4typeEOS9_ = comdat any

$_ZN9__gnu_cxx5__ops14_Iter_comp_valIPFb9ClassProbS2_EEC2EONS0_15_Iter_comp_iterIS4_EE = comdat any

$_ZSt11__push_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElS2_NS0_5__ops14_Iter_comp_valIPFbS2_S2_EEEEvT_T0_SE_T1_RT2_ = comdat any

$_ZSt4moveIRPFb9ClassProbS0_EEONSt16remove_referenceIT_E4typeEOS5_ = comdat any

$_ZN9__gnu_cxx5__ops14_Iter_comp_valIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEES2_EEbT_RT0_ = comdat any

$_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEmmEv = comdat any

$_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_SD_T0_ = comdat any

$_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEmiEl = comdat any

$_ZSt21__unguarded_partitionIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEET_SD_SD_SD_T0_ = comdat any

$_ZSt9iter_swapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES7_EvT_T0_ = comdat any

$_ZSt4swapI9ClassProbENSt9enable_ifIXsr6__and_ISt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SD_ = comdat any

$_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_ = comdat any

$_ZSt26__unguarded_insertion_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_ = comdat any

$_ZN9__gnu_cxxeqIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_ = comdat any

$_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES7_ET0_T_S9_S8_ = comdat any

$_ZSt25__unguarded_linear_insertIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops14_Val_comp_iterIPFbS2_S2_EEEEvT_T0_ = comdat any

$_ZN9__gnu_cxx5__ops15__val_comp_iterIPFb9ClassProbS2_EEENS0_14_Val_comp_iterIT_EENS0_15_Iter_comp_iterIS6_EE = comdat any

$_ZSt23__copy_move_backward_a2ILb1EN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES7_ET1_T0_S9_S8_ = comdat any

$_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEEET_S8_ = comdat any

$_ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_ = comdat any

$_ZSt22__copy_move_backward_aILb1EP9ClassProbS1_ET1_T0_S3_S2_ = comdat any

$_ZSt12__niter_baseIP9ClassProbSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE = comdat any

$_ZNSt20__copy_move_backwardILb1ELb1ESt26random_access_iterator_tagE13__copy_move_bI9ClassProbEEPT_PKS4_S7_S5_ = comdat any

$_ZN9__gnu_cxx5__ops14_Val_comp_iterIPFb9ClassProbS2_EEclIS2_NS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEEEEbRT_T0_ = comdat any

$_ZN9__gnu_cxx5__ops14_Val_comp_iterIPFb9ClassProbS2_EEC2EONS0_15_Iter_comp_iterIS4_EE = comdat any

$_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEC2ES4_ = comdat any

$_ZNSt16allocator_traitsISaIfEE9constructIfJRKfEEEvRS0_PT_DpOT0_ = comdat any

$_ZNSt6vectorIfSaIfEE17_M_realloc_insertIJRKfEEEvN9__gnu_cxx17__normal_iteratorIPfS1_EEDpOT_ = comdat any

$_ZNSt6vectorIfSaIfEE3endEv = comdat any

$_ZN9__gnu_cxx13new_allocatorIfE9constructIfJRKfEEEvPT_DpOT0_ = comdat any

$_ZSt7forwardIRKfEOT_RNSt16remove_referenceIS2_E4typeE = comdat any

$_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc = comdat any

$_ZN9__gnu_cxxmiIPfSt6vectorIfSaIfEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_ = comdat any

$_ZNSt6vectorIfSaIfEE5beginEv = comdat any

$_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm = comdat any

$_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_ = comdat any

$_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv = comdat any

$_ZNKSt6vectorIfSaIfEE8max_sizeEv = comdat any

$_ZNSt6vectorIfSaIfEE11_S_max_sizeERKS0_ = comdat any

$_ZNKSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv = comdat any

$_ZNSt16allocator_traitsISaIfEE8max_sizeERKS0_ = comdat any

$_ZNK9__gnu_cxx13new_allocatorIfE8max_sizeEv = comdat any

$_ZN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEC2ERKS1_ = comdat any

$_ZNSt16allocator_traitsISaIfEE8allocateERS0_m = comdat any

$_ZN9__gnu_cxx13new_allocatorIfE8allocateEmPKv = comdat any

$_ZNSt6vectorIfSaIfEE14_S_do_relocateEPfS2_S2_RS0_St17integral_constantIbLb1EE = comdat any

$_ZSt12__relocate_aIPfS0_SaIfEET0_T_S3_S2_RT1_ = comdat any

$_ZSt14__relocate_a_1IffENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS1_E4typeES2_S2_S2_RSaIT0_E = comdat any

$_ZSt12__niter_baseIPfET_S1_ = comdat any

$_ZStorSt12_Ios_IostateS_ = comdat any

$_ZSt13__check_facetISt5ctypeIcEERKT_PS3_ = comdat any

$_ZNKSt5ctypeIcE5widenEc = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@run_accuracies = dso_local global %"class.std::vector" zeroinitializer, align 8
@_Z17model_params_pathB5cxx11 = dso_local global %"class.std::__cxx11::basic_string" zeroinitializer, align 8
@.str = private unnamed_addr constant [40 x i8] c"../../test/dnn_benchmarks/model_params/\00", align 1
@.str.4 = private unnamed_addr constant [19 x i8] c"tensor dims = %d \0A\00", align 1
@.str.5 = private unnamed_addr constant [18 x i8] c"dim1_size = %lu \0A\00", align 1
@.str.6 = private unnamed_addr constant [18 x i8] c"dim2_size = %lu \0A\00", align 1
@.str.7 = private unnamed_addr constant [18 x i8] c"num_elems = %lu \0A\00", align 1
@.str.8 = private unnamed_addr constant [3 x i8] c"wb\00", align 1
@.str.9 = private unnamed_addr constant [58 x i8] c"File %s could not be created. Check if directory exists \0A\00", align 1
@.str.10 = private unnamed_addr constant [4 x i8] c"%f,\00", align 1
@.str.12 = private unnamed_addr constant [18 x i8] c"Num_elems = %lu \0A\00", align 1
@.str.13 = private unnamed_addr constant [16 x i8] c"dim[%d] = %lu \0A\00", align 1
@.str.14 = private unnamed_addr constant [35 x i8] c"Tensor data mismatch at index %d \0A\00", align 1
@.str.15 = private unnamed_addr constant [21 x i8] c"Tensor data mismatch\00", align 1
@.str.16 = private unnamed_addr constant [3 x i8] c"rb\00", align 1
@.str.17 = private unnamed_addr constant [41 x i8] c"Data file %s is not found. Aborting... \0A\00", align 1
@.str.18 = private unnamed_addr constant [40 x i8] c"Data file %s is not found. Aborting...\0A\00", align 1
@.str.19 = private unnamed_addr constant [37 x i8] c"batch_dim = %lu, num_classes = %lu \0A\00", align 1
@.str.20 = private unnamed_addr constant [24 x i8] c"****** Accuracy = %f \0A\0A\00", align 1
@.str.21 = private unnamed_addr constant [15 x i8] c"final_accuracy\00", align 1
@.str.22 = private unnamed_addr constant [3 x i8] c"w+\00", align 1
@.str.23 = private unnamed_addr constant [34 x i8] c"batch_dim = %lu, channels = %lu \0A\00", align 1
@.str.24 = private unnamed_addr constant [30 x i8] c"\0A\0A **** Final Accuracy = %f \0A\00", align 1
@.str.25 = private unnamed_addr constant [9 x i8] c"avg_psnr\00", align 1
@.str.26 = private unnamed_addr constant [13 x i8] c"psnr_std.txt\00", align 1
@.str.27 = private unnamed_addr constant [19 x i8] c"run_accuracies.txt\00", align 1
@.str.28 = private unnamed_addr constant [2 x i8] c"r\00", align 1
@.str.30 = private unnamed_addr constant [3 x i8] c"%f\00", align 1
@.str.31 = private unnamed_addr constant [23 x i8] c"**** PSNR read = %f \0A\0A\00", align 1
@.str.32 = private unnamed_addr constant [9 x i8] c"psnr.txt\00", align 1
@.str.33 = private unnamed_addr constant [36 x i8] c"batch_dim = %lu, image_size = %lu \0A\00", align 1
@.str.34 = private unnamed_addr constant [13 x i8] c"img_psnr.txt\00", align 1
@.str.35 = private unnamed_addr constant [18 x i8] c"PSNR value = %f \0A\00", align 1
@.str.36 = private unnamed_addr constant [26 x i8] c"*** violation_rate= %f \0A\0A\00", align 1
@.str.37 = private unnamed_addr constant [22 x i8] c"*** avg_psnr =  %f \0A\0A\00", align 1
@.str.38 = private unnamed_addr constant [23 x i8] c"** Output size = %lu \0A\00", align 1
@_ZSt4cout = external dso_local global %"class.std::basic_ostream", align 8
@.str.39 = private unnamed_addr constant [15 x i8] c"args pointer: \00", align 1
@.str.40 = private unnamed_addr constant [15 x i8] c"arg1 pointer: \00", align 1
@.str.41 = private unnamed_addr constant [15 x i8] c"arg2 pointer: \00", align 1
@.str.42 = private unnamed_addr constant [11 x i8] c"num_elems \00", align 1
@.str.43 = private unnamed_addr constant [14 x i8] c"size_in_bytes\00", align 1
@.str.44 = private unnamed_addr constant [18 x i8] c"Output of tensor \00", align 1
@.str.45 = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.46 = private unnamed_addr constant [42 x i8] c"basic_string::_M_construct null not valid\00", align 1
@_ZTVNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE = external dso_local unnamed_addr constant { [5 x i8*], [5 x i8*] }, align 8
@_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE = external unnamed_addr constant [4 x i8*], align 8
@_ZTVSt9basic_iosIcSt11char_traitsIcEE = external dso_local unnamed_addr constant { [4 x i8*] }, align 8
@_ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE = external dso_local unnamed_addr constant { [16 x i8*] }, align 8
@_ZTVSt15basic_streambufIcSt11char_traitsIcEE = external dso_local unnamed_addr constant { [16 x i8*] }, align 8
@.str.47 = private unnamed_addr constant [22 x i8] c"basic_string::replace\00", align 1
@.str.48 = private unnamed_addr constant [55 x i8] c"%s: __pos (which is %zu) > this->size() (which is %zu)\00", align 1
@.str.49 = private unnamed_addr constant [26 x i8] c"vector::_M_realloc_insert\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_simple_graph.cpp, i8* null }]
@str = private unnamed_addr constant [23 x i8] c"Successful cudaMalloc \00", align 1
@str.50 = private unnamed_addr constant [28 x i8] c"ERROR: psnr.txt not found! \00", align 1
@0 = constant [21 x i8] c"data/tuner_confs.txt\00"
@1 = constant [2 x i8] c"1\00"
@2 = constant [2 x i8] c"2\00"
@3 = constant [2 x i8] c"3\00"

; Function Attrs: uwtable
define internal fastcc void @__cxx_global_var_init() unnamed_addr #0 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %0 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #20
  ret void
}

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) unnamed_addr #1

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) unnamed_addr #2

; Function Attrs: nofree nounwind
declare dso_local i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #3

; Function Attrs: nounwind uwtable
define internal fastcc void @__cxx_global_var_init.1() unnamed_addr #4 section ".text.startup" {
entry:
  tail call void @_ZNSt6vectorIfSaIfEEC2Ev(%"class.std::vector"* nonnull @run_accuracies) #20
  %0 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::vector"*)* @_ZNSt6vectorIfSaIfEED2Ev to void (i8*)*), i8* bitcast (%"class.std::vector"* @run_accuracies to i8*), i8* nonnull @__dso_handle) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6vectorIfSaIfEEC2Ev(%"class.std::vector"* %this) unnamed_addr #4 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0
  tail call void @_ZNSt12_Vector_baseIfSaIfEEC2Ev(%"struct.std::_Vector_base"* %0) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6vectorIfSaIfEED2Ev(%"class.std::vector"* %this) unnamed_addr #4 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %0 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0
  %_M_start = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 0
  %1 = load float*, float** %_M_start, align 8, !tbaa !6
  %_M_finish = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 1
  %2 = load float*, float** %_M_finish, align 8, !tbaa !11
  %call = tail call dereferenceable(1) %"class.std::allocator"* @_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base"* %0) #20
  invoke void @_ZSt8_DestroyIPffEvT_S1_RSaIT0_E(float* %1, float* %2, %"class.std::allocator"* nonnull dereferenceable(1) %call)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  tail call void @_ZNSt12_Vector_baseIfSaIfEED2Ev(%"struct.std::_Vector_base"* %0) #20
  ret void

lpad:                                             ; preds = %entry
  %3 = landingpad { i8*, i32 }
          catch i8* null
  %4 = extractvalue { i8*, i32 } %3, 0
  tail call void @_ZNSt12_Vector_baseIfSaIfEED2Ev(%"struct.std::_Vector_base"* %0) #20
  tail call void @__clang_call_terminate(i8* %4) #21
  unreachable
}

; Function Attrs: uwtable
define internal fastcc void @__cxx_global_var_init.2() unnamed_addr #0 section ".text.startup" personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %ref.tmp = alloca %"class.std::allocator.0", align 1
  %0 = getelementptr inbounds %"class.std::allocator.0", %"class.std::allocator.0"* %ref.tmp, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %0) #20
  call void @_ZNSaIcEC2Ev(%"class.std::allocator.0"* nonnull %ref.tmp) #20
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_(%"class.std::__cxx11::basic_string"* nonnull @_Z17model_params_pathB5cxx11, i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str, i64 0, i64 0), %"class.std::allocator.0"* nonnull dereferenceable(1) %ref.tmp)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  call void @_ZNSaIcED2Ev(%"class.std::allocator.0"* nonnull %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %0) #20
  %1 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::__cxx11::basic_string"*)* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev to void (i8*)*), i8* bitcast (%"class.std::__cxx11::basic_string"* @_Z17model_params_pathB5cxx11 to i8*), i8* nonnull @__dso_handle) #20
  ret void

lpad:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 }
          cleanup
  call void @_ZNSaIcED2Ev(%"class.std::allocator.0"* nonnull %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %0) #20
  resume { i8*, i32 } %2
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSaIcEC2Ev(%"class.std::allocator.0"* %this) unnamed_addr #4 align 2 {
entry:
  %0 = bitcast %"class.std::allocator.0"* %this to %"class.__gnu_cxx::new_allocator.1"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIcEC2Ev(%"class.__gnu_cxx::new_allocator.1"* %0) #20
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2EPKcRKS3_(%"class.std::__cxx11::basic_string"* %this, i8* %__s, %"class.std::allocator.0"* dereferenceable(1) %__a) unnamed_addr #0 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %_M_dataplus = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 0
  %call = tail call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv(%"class.std::__cxx11::basic_string"* %this)
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC2EPcRKS3_(%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %_M_dataplus, i8* %call, %"class.std::allocator.0"* nonnull dereferenceable(1) %__a)
  %tobool = icmp eq i8* %__s, null
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %call2 = tail call i64 @_ZNSt11char_traitsIcE6lengthEPKc(i8* nonnull %__s)
  %add.ptr = getelementptr inbounds i8, i8* %__s, i64 %call2
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %entry
  %cond = phi i8* [ %add.ptr, %cond.true ], [ inttoptr (i64 -1 to i8*), %entry ]
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_(%"class.std::__cxx11::basic_string"* %this, i8* %__s, i8* %cond)
          to label %invoke.cont4 unwind label %lpad

invoke.cont4:                                     ; preds = %cond.end
  ret void

lpad:                                             ; preds = %cond.end
  %0 = landingpad { i8*, i32 }
          cleanup
  %1 = bitcast %"class.std::__cxx11::basic_string"* %this to %"class.std::allocator.0"*
  tail call void @_ZNSaIcED2Ev(%"class.std::allocator.0"* %1) #20
  resume { i8*, i32 } %0
}

declare dso_local i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #5

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* %this) unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv(%"class.std::__cxx11::basic_string"* %this)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %0 = bitcast %"class.std::__cxx11::basic_string"* %this to %"class.std::allocator.0"*
  tail call void @_ZNSaIcED2Ev(%"class.std::allocator.0"* %0) #20
  ret void

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = bitcast %"class.std::__cxx11::basic_string"* %this to %"class.std::allocator.0"*
  tail call void @_ZNSaIcED2Ev(%"class.std::allocator.0"* %3) #20
  tail call void @__clang_call_terminate(i8* %2) #21
  unreachable
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @_Z15printTensorInfoPv(i8* nocapture readonly %tensor_ptr) local_unnamed_addr #6 {
entry:
  %gpu_data = getelementptr inbounds i8, i8* %tensor_ptr, i64 56
  %0 = bitcast i8* %gpu_data to i8**
  %1 = load i8*, i8** %0, align 8, !tbaa !12
  %cmp = icmp eq i8* %1, null
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @str, i64 0, i64 0))
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %dims = getelementptr inbounds i8, i8* %tensor_ptr, i64 88
  %num_dims = bitcast i8* %dims to i32*
  %2 = load i32, i32* %num_dims, align 8, !tbaa !18
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.4, i64 0, i64 0), i32 %2)
  %dim_sizes = getelementptr inbounds i8, i8* %tensor_ptr, i64 96
  %3 = bitcast i8* %dim_sizes to i64**
  %4 = load i64*, i64** %3, align 8, !tbaa !19
  %5 = load i64, i64* %4, align 8, !tbaa !20
  %call3 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.5, i64 0, i64 0), i64 %5)
  %6 = load i64*, i64** %3, align 8, !tbaa !19
  %arrayidx6 = getelementptr inbounds i64, i64* %6, i64 1
  %7 = load i64, i64* %arrayidx6, align 8, !tbaa !20
  %call7 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.6, i64 0, i64 0), i64 %7)
  %num_elems = getelementptr inbounds i8, i8* %tensor_ptr, i64 72
  %8 = bitcast i8* %num_elems to i64*
  %9 = load i64, i64* %8, align 8, !tbaa !21
  %call8 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.7, i64 0, i64 0), i64 %9)
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #7

; Function Attrs: uwtable
define dso_local void @_Z17dumpWeightsToFilePcPv(i8* %file_name, i8* %weights_ptr) local_unnamed_addr #0 {
entry:
  tail call void @hpvm_request_tensor(i8* %weights_ptr, i32 0)
  %call = tail call %struct._IO_FILE* @fopen(i8* %file_name, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.8, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([58 x i8], [58 x i8]* @.str.9, i64 0, i64 0), i8* %file_name)
  tail call void @abort() #21
  unreachable

if.end:                                           ; preds = %entry
  %host_data = getelementptr inbounds i8, i8* %weights_ptr, i64 48
  %0 = bitcast i8* %host_data to i8**
  %1 = load i8*, i8** %0, align 8, !tbaa !22
  %size_in_bytes = getelementptr inbounds i8, i8* %weights_ptr, i64 80
  %2 = bitcast i8* %size_in_bytes to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !23
  %4 = tail call i64 @fwrite_unlocked(i8* %1, i64 1, i64 %3, %struct._IO_FILE* nonnull %call)
  %call3 = tail call i32 @fclose(%struct._IO_FILE* nonnull %call)
  ret void
}

declare dso_local void @hpvm_request_tensor(i8*, i32) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare dso_local noalias %struct._IO_FILE* @fopen(i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #7

; Function Attrs: noreturn nounwind
declare dso_local void @abort() local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare dso_local i32 @fclose(%struct._IO_FILE* nocapture) local_unnamed_addr #7

; Function Attrs: uwtable
define dso_local void @_Z18fillTensorWithOnesPv(i8* %tensor_ptr) local_unnamed_addr #0 {
entry:
  tail call void @hpvm_request_tensor(i8* %tensor_ptr, i32 0)
  %data_type = bitcast i8* %tensor_ptr to i32*
  %0 = load i32, i32* %data_type, align 8, !tbaa !24
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %host_data = getelementptr inbounds i8, i8* %tensor_ptr, i64 48
  %1 = bitcast i8* %host_data to float**
  %2 = load float*, float** %1, align 8, !tbaa !22
  %num_elems = getelementptr inbounds i8, i8* %tensor_ptr, i64 72
  %3 = bitcast i8* %num_elems to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !21
  %cmp110 = icmp eq i64 %4, 0
  br i1 %cmp110, label %if.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.then
  %5 = load i64, i64* %3, align 8, !tbaa !21
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %conv12 = phi i64 [ 0, %for.body.lr.ph ], [ %conv, %for.body ]
  %i.011 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %2, i64 %conv12
  store float 1.000000e+00, float* %arrayidx, align 4, !tbaa !25
  %inc = add i32 %i.011, 1
  %conv = zext i32 %inc to i64
  %cmp1 = icmp ugt i64 %5, %conv
  br i1 %cmp1, label %for.body, label %if.end

if.end:                                           ; preds = %for.body, %if.then, %entry
  ret void
}

; Function Attrs: uwtable
define dso_local void @_Z19fillWithOnesAndTwosPv(i8* %tensor_ptr) local_unnamed_addr #0 {
entry:
  tail call void @hpvm_request_tensor(i8* %tensor_ptr, i32 0)
  %data_type = bitcast i8* %tensor_ptr to i32*
  %0 = load i32, i32* %data_type, align 8, !tbaa !24
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %host_data = getelementptr inbounds i8, i8* %tensor_ptr, i64 48
  %1 = bitcast i8* %host_data to float**
  %2 = load float*, float** %1, align 8, !tbaa !22
  %num_elems = getelementptr inbounds i8, i8* %tensor_ptr, i64 72
  %3 = bitcast i8* %num_elems to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !21
  %cmp136 = icmp ult i64 %4, 2
  br i1 %cmp136, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.then
  %5 = load i64, i64* %3, align 8, !tbaa !21
  %div = lshr i64 %5, 1
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %if.then
  %.lcssa = phi i64 [ %4, %if.then ], [ %5, %for.body ]
  %div.lcssa = phi i64 [ 0, %if.then ], [ %div, %for.body ]
  %conv731 = and i64 %div.lcssa, 4294967295
  %cmp932 = icmp ugt i64 %.lcssa, %conv731
  br i1 %cmp932, label %for.body11.preheader, label %if.end

for.body11.preheader:                             ; preds = %for.cond.cleanup
  %conv5 = trunc i64 %div.lcssa to i32
  br label %for.body11

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %conv38 = phi i64 [ 0, %for.body.lr.ph ], [ %conv, %for.body ]
  %i.037 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %2, i64 %conv38
  store float 1.000000e+00, float* %arrayidx, align 4, !tbaa !25
  %inc = add i32 %i.037, 1
  %conv = zext i32 %inc to i64
  %cmp1 = icmp ugt i64 %div, %conv
  br i1 %cmp1, label %for.body, label %for.cond.cleanup

for.body11:                                       ; preds = %for.body11, %for.body11.preheader
  %conv734 = phi i64 [ %conv7, %for.body11 ], [ %conv731, %for.body11.preheader ]
  %i2.033 = phi i32 [ %inc15, %for.body11 ], [ %conv5, %for.body11.preheader ]
  %arrayidx13 = getelementptr inbounds float, float* %2, i64 %conv734
  store float 2.000000e+00, float* %arrayidx13, align 4, !tbaa !25
  %inc15 = add i32 %i2.033, 1
  %conv7 = zext i32 %inc15 to i64
  %cmp9 = icmp ugt i64 %.lcssa, %conv7
  br i1 %cmp9, label %for.body11, label %if.end

if.end:                                           ; preds = %for.body11, %for.cond.cleanup, %entry
  ret void
}

; Function Attrs: uwtable
define dso_local void @_Z17fillTensorWithValPvf(i8* %tensor_ptr, float %target_value) local_unnamed_addr #0 {
entry:
  tail call void @hpvm_request_tensor(i8* %tensor_ptr, i32 0)
  %data_type = bitcast i8* %tensor_ptr to i32*
  %0 = load i32, i32* %data_type, align 8, !tbaa !24
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %host_data = getelementptr inbounds i8, i8* %tensor_ptr, i64 48
  %1 = bitcast i8* %host_data to float**
  %2 = load float*, float** %1, align 8, !tbaa !22
  %num_elems = getelementptr inbounds i8, i8* %tensor_ptr, i64 72
  %3 = bitcast i8* %num_elems to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !21
  %cmp110 = icmp eq i64 %4, 0
  br i1 %cmp110, label %if.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.then
  %5 = load i64, i64* %3, align 8, !tbaa !21
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %conv12 = phi i64 [ 0, %for.body.lr.ph ], [ %conv, %for.body ]
  %i.011 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %2, i64 %conv12
  store float %target_value, float* %arrayidx, align 4, !tbaa !25
  %inc = add i32 %i.011, 1
  %conv = zext i32 %inc to i64
  %cmp1 = icmp ugt i64 %5, %conv
  br i1 %cmp1, label %for.body, label %if.end

if.end:                                           ; preds = %for.body, %if.then, %entry
  ret void
}

; Function Attrs: uwtable
define dso_local void @_Z21fillTensorWithNegOnesPv(i8* %tensor_ptr) local_unnamed_addr #0 {
entry:
  tail call void @hpvm_request_tensor(i8* %tensor_ptr, i32 0)
  %data_type = bitcast i8* %tensor_ptr to i32*
  %0 = load i32, i32* %data_type, align 8, !tbaa !24
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %host_data = getelementptr inbounds i8, i8* %tensor_ptr, i64 48
  %1 = bitcast i8* %host_data to float**
  %2 = load float*, float** %1, align 8, !tbaa !22
  %num_elems = getelementptr inbounds i8, i8* %tensor_ptr, i64 72
  %3 = bitcast i8* %num_elems to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !21
  %cmp110 = icmp eq i64 %4, 0
  br i1 %cmp110, label %if.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.then
  %5 = load i64, i64* %3, align 8, !tbaa !21
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %conv12 = phi i64 [ 0, %for.body.lr.ph ], [ %conv, %for.body ]
  %i.011 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %2, i64 %conv12
  store float -1.000000e+00, float* %arrayidx, align 4, !tbaa !25
  %inc = add i32 %i.011, 1
  %conv = zext i32 %inc to i64
  %cmp1 = icmp ugt i64 %5, %conv
  br i1 %cmp1, label %for.body, label %if.end

if.end:                                           ; preds = %for.body, %if.then, %entry
  ret void
}

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local void @_Z14fillTensorValsPv(i8* nocapture readonly %tensor_ptr) local_unnamed_addr #9 {
entry:
  %data_type = bitcast i8* %tensor_ptr to i32*
  %0 = load i32, i32* %data_type, align 8, !tbaa !24
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %host_data = getelementptr inbounds i8, i8* %tensor_ptr, i64 48
  %1 = bitcast i8* %host_data to float**
  %2 = load float*, float** %1, align 8, !tbaa !22
  %num_elems = getelementptr inbounds i8, i8* %tensor_ptr, i64 72
  %3 = bitcast i8* %num_elems to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !21
  %cmp111 = icmp eq i64 %4, 0
  br i1 %cmp111, label %if.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %if.then
  %5 = load i64, i64* %3, align 8, !tbaa !21
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %conv13 = phi i64 [ 0, %for.body.lr.ph ], [ %conv, %for.body ]
  %i.012 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %for.body ]
  %add = add i32 %i.012, 1
  %conv2 = uitofp i32 %add to float
  %arrayidx = getelementptr inbounds float, float* %2, i64 %conv13
  store float %conv2, float* %arrayidx, align 4, !tbaa !25
  %conv = zext i32 %add to i64
  %cmp1 = icmp ugt i64 %5, %conv
  br i1 %cmp1, label %for.body, label %if.end

if.end:                                           ; preds = %for.body, %if.then, %entry
  ret void
}

; Function Attrs: uwtable
define dso_local void @_Z17printTensorValuesPv(i8* %tensor_ptr) local_unnamed_addr #0 {
entry:
  tail call void @hpvm_request_tensor(i8* %tensor_ptr, i32 0)
  %data_type = bitcast i8* %tensor_ptr to i32*
  %0 = load i32, i32* %data_type, align 8, !tbaa !24
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %host_data = getelementptr inbounds i8, i8* %tensor_ptr, i64 48
  %1 = bitcast i8* %host_data to float**
  %2 = load float*, float** %1, align 8, !tbaa !22
  %num_elems = getelementptr inbounds i8, i8* %tensor_ptr, i64 72
  %3 = bitcast i8* %num_elems to i64*
  %4 = load i64, i64* %3, align 8, !tbaa !21
  %cmp112 = icmp eq i64 %4, 0
  br i1 %cmp112, label %if.end, label %for.body

for.body:                                         ; preds = %for.body, %if.then
  %conv14 = phi i64 [ %conv, %for.body ], [ 0, %if.then ]
  %i.013 = phi i32 [ %inc, %for.body ], [ 0, %if.then ]
  %arrayidx = getelementptr inbounds float, float* %2, i64 %conv14
  %5 = load float, float* %arrayidx, align 4, !tbaa !25
  %conv2 = fpext float %5 to double
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.10, i64 0, i64 0), double %conv2)
  %inc = add i32 %i.013, 1
  %conv = zext i32 %inc to i64
  %6 = load i64, i64* %3, align 8, !tbaa !21
  %cmp1 = icmp ugt i64 %6, %conv
  br i1 %cmp1, label %for.body, label %if.end

if.end:                                           ; preds = %for.body, %if.then, %entry
  %putchar = tail call i32 @putchar(i32 10)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @_Z15printTensorDimsPv(i8* nocapture readonly %tensor_ptr) local_unnamed_addr #6 {
entry:
  %num_elems = getelementptr inbounds i8, i8* %tensor_ptr, i64 72
  %0 = bitcast i8* %num_elems to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !21
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.12, i64 0, i64 0), i64 %1)
  %dims = getelementptr inbounds i8, i8* %tensor_ptr, i64 88
  %num_dims = bitcast i8* %dims to i32*
  %2 = load i32, i32* %num_dims, align 8, !tbaa !18
  %cmp10 = icmp eq i32 %2, 0
  br i1 %cmp10, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %dim_sizes = getelementptr inbounds i8, i8* %tensor_ptr, i64 96
  %3 = bitcast i8* %dim_sizes to i64**
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %4 = load i64*, i64** %3, align 8, !tbaa !19
  %arrayidx = getelementptr inbounds i64, i64* %4, i64 %indvars.iv
  %5 = load i64, i64* %arrayidx, align 8, !tbaa !20
  %6 = trunc i64 %indvars.iv to i32
  %call2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.13, i64 0, i64 0), i32 %6, i64 %5)
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %7 = load i32, i32* %num_dims, align 8, !tbaa !18
  %8 = zext i32 %7 to i64
  %cmp = icmp ult i64 %indvars.iv.next, %8
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Function Attrs: uwtable
define dso_local void @_Z14compareTensorsPvS_(i8* %tensor1_ptr, i8* %tensor2_ptr) local_unnamed_addr #0 {
entry:
  tail call void @hpvm_request_tensor(i8* %tensor1_ptr, i32 0)
  tail call void @hpvm_request_tensor(i8* %tensor2_ptr, i32 0)
  %host_data = getelementptr inbounds i8, i8* %tensor1_ptr, i64 48
  %0 = bitcast i8* %host_data to float**
  %1 = load float*, float** %0, align 8, !tbaa !22
  %host_data1 = getelementptr inbounds i8, i8* %tensor2_ptr, i64 48
  %2 = bitcast i8* %host_data1 to float**
  %3 = load float*, float** %2, align 8, !tbaa !22
  %num_elems = getelementptr inbounds i8, i8* %tensor1_ptr, i64 72
  %4 = bitcast i8* %num_elems to i64*
  %5 = load i64, i64* %4, align 8, !tbaa !21
  %cmp17 = icmp eq i64 %5, 0
  br i1 %cmp17, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void

for.body:                                         ; preds = %for.inc, %entry
  %conv19 = phi i64 [ %conv, %for.inc ], [ 0, %entry ]
  %i.018 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %1, i64 %conv19
  %6 = load float, float* %arrayidx, align 4, !tbaa !25
  %arrayidx3 = getelementptr inbounds float, float* %3, i64 %conv19
  %7 = load float, float* %arrayidx3, align 4, !tbaa !25
  %cmp4 = fcmp une float %6, %7
  br i1 %cmp4, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.14, i64 0, i64 0), i32 %i.018)
  tail call void @abort() #21
  unreachable

for.inc:                                          ; preds = %for.body
  %inc = add i32 %i.018, 1
  %conv = zext i32 %inc to i64
  %8 = load i64, i64* %4, align 8, !tbaa !21
  %cmp = icmp ugt i64 %8, %conv
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; Function Attrs: uwtable
define dso_local void @_Z13compareValuesPvPfm(i8* %tensor_ptr, float* nocapture readonly %data, i64 %num_elems) local_unnamed_addr #0 {
entry:
  tail call void @hpvm_request_tensor(i8* %tensor_ptr, i32 0)
  %host_data = getelementptr inbounds i8, i8* %tensor_ptr, i64 48
  %0 = bitcast i8* %host_data to float**
  %1 = load float*, float** %0, align 8, !tbaa !22
  %cmp11 = icmp eq i64 %num_elems, 0
  br i1 %cmp11, label %for.cond.cleanup, label %for.body

for.cond:                                         ; preds = %for.body
  %conv = zext i32 %inc to i64
  %cmp = icmp ult i64 %conv, %num_elems
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond, %entry
  ret void

for.body:                                         ; preds = %for.cond, %entry
  %conv13 = phi i64 [ %conv, %for.cond ], [ 0, %entry ]
  %i.012 = phi i32 [ %inc, %for.cond ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds float, float* %1, i64 %conv13
  %2 = load float, float* %arrayidx, align 4, !tbaa !25
  %arrayidx2 = getelementptr inbounds float, float* %data, i64 %conv13
  %3 = load float, float* %arrayidx2, align 4, !tbaa !25
  %cmp3 = fcmp une float %2, %3
  %inc = add i32 %i.012, 1
  br i1 %cmp3, label %if.then, label %for.cond

if.then:                                          ; preds = %for.body
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.15, i64 0, i64 0))
  tail call void @abort() #21
  unreachable
}

; Function Attrs: uwtable
define dso_local %struct.Tensor* @_Z18readTrainedWeightsPKcillll(i8* %file_name, i32 %data_type, i64 %dim1_size, i64 %dim2_size, i64 %dim3_size, i64 %dim4_size) local_unnamed_addr #0 {
entry:
  %mul = mul i64 %dim3_size, %dim2_size
  %mul3 = mul i64 %dim3_size, %dim2_size
  %mul4 = shl i64 %mul3, 2
  %mul5 = mul i64 %mul4, %dim1_size
  %mul6 = mul i64 %mul5, %dim4_size
  %mul1 = shl i64 %mul, 2
  %mul2 = mul i64 %mul1, %dim1_size
  %mul7 = mul i64 %mul2, %dim4_size
  %call = tail call noalias i8* @malloc(i64 %mul7) #20
  %call8 = tail call %struct._IO_FILE* @fopen(i8* %file_name, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.16, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call8, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call9 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([41 x i8], [41 x i8]* @.str.17, i64 0, i64 0), i8* %file_name)
  tail call void @abort() #21
  unreachable

if.end:                                           ; preds = %entry
  %call11 = tail call i32 @fseek(%struct._IO_FILE* nonnull %call8, i64 0, i32 1)
  %0 = tail call i64 @fread_unlocked(i8* %call, i64 1, i64 %mul6, %struct._IO_FILE* nonnull %call8)
  %call13 = tail call i32 @fclose(%struct._IO_FILE* nonnull %call8)
  %call14 = tail call i8* @create4DTensor(i32 %data_type, i32 0, i64 %dim1_size, i64 %dim2_size, i64 %dim3_size, i64 %dim4_size)
  %1 = bitcast i8* %call14 to %struct.Tensor*
  tail call void @initTensorData(i8* %call14, i8* %call, i64 %mul6)
  tail call void @free(i8* %call) #20
  ret %struct.Tensor* %1
}

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #7

; Function Attrs: nofree nounwind
declare dso_local i32 @fseek(%struct._IO_FILE* nocapture, i64, i32) local_unnamed_addr #7

declare dso_local i8* @create4DTensor(i32, i32, i64, i64, i64, i64) local_unnamed_addr #1

declare dso_local void @initTensorData(i8*, i8*, i64) local_unnamed_addr #1

; Function Attrs: nounwind
declare dso_local void @free(i8* nocapture) local_unnamed_addr #2

; Function Attrs: uwtable
define dso_local %struct.Tensor* @_Z14readInputBatchPKcllllll(i8* %file_name, i64 %data_type, i64 %start, i64 %end, i64 %dim2_size, i64 %dim3_size, i64 %dim4_size) local_unnamed_addr #0 {
entry:
  %sub = sub nsw i64 %end, %start
  %mul = mul i64 %dim3_size, %dim2_size
  %mul3 = mul i64 %dim3_size, %dim2_size
  %mul4 = shl i64 %mul3, 2
  %mul5 = mul i64 %mul4, %sub
  %mul6 = mul i64 %mul5, %dim4_size
  %mul1 = shl i64 %mul, 2
  %mul2 = mul i64 %mul1, %sub
  %mul7 = mul i64 %mul2, %dim4_size
  %call = tail call noalias i8* @malloc(i64 %mul7) #20
  %call12 = tail call %struct._IO_FILE* @fopen(i8* %file_name, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.16, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call12, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call13 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([41 x i8], [41 x i8]* @.str.17, i64 0, i64 0), i8* %file_name)
  tail call void @abort() #21
  unreachable

if.end:                                           ; preds = %entry
  %mul8 = mul i64 %dim3_size, %dim2_size
  %mul9 = shl i64 %mul8, 2
  %mul10 = mul i64 %mul9, %start
  %mul11 = mul i64 %mul10, %dim4_size
  %call14 = tail call i32 @fseek(%struct._IO_FILE* nonnull %call12, i64 %mul11, i32 0)
  %0 = tail call i64 @fread_unlocked(i8* %call, i64 1, i64 %mul6, %struct._IO_FILE* nonnull %call12)
  %call16 = tail call i32 @fclose(%struct._IO_FILE* nonnull %call12)
  %conv = trunc i64 %data_type to i32
  %call17 = tail call i8* @create4DTensor(i32 %conv, i32 0, i64 %sub, i64 %dim2_size, i64 %dim3_size, i64 %dim4_size)
  %1 = bitcast i8* %call17 to %struct.Tensor*
  tail call void @initTensorData(i8* %call17, i8* %call, i64 %mul6)
  tail call void @free(i8* %call) #20
  ret %struct.Tensor* %1
}

; Function Attrs: nounwind uwtable
define dso_local noalias i8* @_Z10readLabelsPKci(i8* %labels_file, i32 %num_labels) local_unnamed_addr #4 {
entry:
  %conv = sext i32 %num_labels to i64
  %call = tail call noalias i8* @malloc(i64 %conv) #20
  %call1 = tail call %struct._IO_FILE* @fopen(i8* %labels_file, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.16, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call1, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.18, i64 0, i64 0), i8* %labels_file)
  tail call void @abort() #21
  unreachable

if.end:                                           ; preds = %entry
  %0 = tail call i64 @fread_unlocked(i8* %call, i64 1, i64 %conv, %struct._IO_FILE* nonnull %call1)
  %call6 = tail call i32 @fclose(%struct._IO_FILE* nonnull %call1)
  ret i8* %call
}

; Function Attrs: nounwind uwtable
define dso_local noalias i32* @_Z11readLabels3PKci(i8* %labels_file, i32 %num_labels) local_unnamed_addr #4 {
entry:
  %conv = sext i32 %num_labels to i64
  %mul = shl nsw i64 %conv, 2
  %call = tail call noalias i8* @malloc(i64 %mul) #20
  %call1 = tail call %struct._IO_FILE* @fopen(i8* %labels_file, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.16, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call1, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.18, i64 0, i64 0), i8* %labels_file)
  tail call void @abort() #21
  unreachable

if.end:                                           ; preds = %entry
  %0 = bitcast i8* %call to i32*
  %1 = tail call i64 @fread_unlocked(i8* %call, i64 1, i64 %mul, %struct._IO_FILE* nonnull %call1)
  %call6 = tail call i32 @fclose(%struct._IO_FILE* nonnull %call1)
  ret i32* %0
}

; Function Attrs: nounwind uwtable
define dso_local noalias i32* @_Z16readLabelsBatch3PKcii(i8* %labels_file, i32 %start, i32 %end) local_unnamed_addr #4 {
entry:
  %sub = sub nsw i32 %end, %start
  %conv2 = sext i32 %sub to i64
  %mul3 = shl nsw i64 %conv2, 2
  %call = tail call noalias i8* @malloc(i64 %mul3) #20
  %call4 = tail call %struct._IO_FILE* @fopen(i8* %labels_file, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.16, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call4, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call5 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.18, i64 0, i64 0), i8* %labels_file)
  tail call void @abort() #21
  unreachable

if.end:                                           ; preds = %entry
  %0 = bitcast i8* %call to i32*
  %mul = shl i32 %start, 2
  %conv6 = sext i32 %mul to i64
  %call7 = tail call i32 @fseek(%struct._IO_FILE* nonnull %call4, i64 %conv6, i32 0)
  %1 = tail call i64 @fread_unlocked(i8* %call, i64 1, i64 %mul3, %struct._IO_FILE* nonnull %call4)
  %call11 = tail call i32 @fclose(%struct._IO_FILE* nonnull %call4)
  ret i32* %0
}

; Function Attrs: uwtable
define dso_local float @_Z16computeAccuracy3PjPv(i32* nocapture readonly %labels, i8* nocapture readonly %result_ptr) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %ss = alloca %"class.std::__cxx11::basic_ostringstream", align 8
  %print_str = alloca %"class.std::__cxx11::basic_string", align 8
  %dim_sizes = getelementptr inbounds i8, i8* %result_ptr, i64 96
  %0 = bitcast i8* %dim_sizes to i64**
  %1 = load i64*, i64** %0, align 8, !tbaa !19
  %2 = load i64, i64* %1, align 8, !tbaa !20
  %arrayidx3 = getelementptr inbounds i64, i64* %1, i64 1
  %3 = load i64, i64* %arrayidx3, align 8, !tbaa !20
  %host_data = getelementptr inbounds i8, i8* %result_ptr, i64 48
  %4 = bitcast i8* %host_data to float**
  %5 = load float*, float** %4, align 8, !tbaa !22
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.19, i64 0, i64 0), i64 %2, i64 %3)
  %cmp104 = icmp eq i64 %2, 0
  br i1 %cmp104, label %for.cond.cleanup, label %for.cond4.preheader.lr.ph

for.cond4.preheader.lr.ph:                        ; preds = %entry
  %cmp6100 = icmp ugt i64 %3, 1
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.cond.cleanup7, %for.cond4.preheader.lr.ph
  %conv107 = phi i64 [ 0, %for.cond4.preheader.lr.ph ], [ %conv, %for.cond.cleanup7 ]
  %num_errors.0106 = phi i32 [ 0, %for.cond4.preheader.lr.ph ], [ %spec.select, %for.cond.cleanup7 ]
  %i.0105 = phi i32 [ 0, %for.cond4.preheader.lr.ph ], [ %inc24, %for.cond.cleanup7 ]
  br i1 %cmp6100, label %for.body8.lr.ph, label %for.cond.cleanup7

for.body8.lr.ph:                                  ; preds = %for.cond4.preheader
  %mul = mul i64 %conv107, %3
  br label %for.body8

for.cond.cleanup.loopexit:                        ; preds = %for.cond.cleanup7
  %phitmp = zext i32 %spec.select to i64
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %num_errors.0.lcssa = phi i64 [ 0, %entry ], [ %phitmp, %for.cond.cleanup.loopexit ]
  %sub = sub i64 %2, %num_errors.0.lcssa
  %conv27 = uitofp i64 %sub to double
  %conv29 = uitofp i64 %2 to double
  %div = fdiv double %conv27, %conv29
  %mul31 = fmul double %div, 1.000000e+02
  %conv32 = fptrunc double %mul31 to float
  %conv33 = fpext float %conv32 to double
  %call34 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.20, i64 0, i64 0), double %conv33)
  %call35 = tail call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.22, i64 0, i64 0))
  %cmp36 = icmp eq %struct._IO_FILE* %call35, null
  br i1 %cmp36, label %if.end50, label %if.then37

for.cond.cleanup7:                                ; preds = %for.body8, %for.cond4.preheader
  %chosen.0.lcssa = phi i32 [ 0, %for.cond4.preheader ], [ %chosen.1, %for.body8 ]
  %arrayidx18 = getelementptr inbounds i32, i32* %labels, i64 %conv107
  %6 = load i32, i32* %arrayidx18, align 4, !tbaa !27
  %cmp19 = icmp ne i32 %chosen.0.lcssa, %6
  %inc21 = zext i1 %cmp19 to i32
  %spec.select = add nuw nsw i32 %num_errors.0106, %inc21
  %inc24 = add i32 %i.0105, 1
  %conv = zext i32 %inc24 to i64
  %cmp = icmp ugt i64 %2, %conv
  br i1 %cmp, label %for.cond4.preheader, label %for.cond.cleanup.loopexit

for.body8:                                        ; preds = %for.body8, %for.body8.lr.ph
  %conv5103 = phi i64 [ 1, %for.body8.lr.ph ], [ %conv5, %for.body8 ]
  %id.0102 = phi i32 [ 1, %for.body8.lr.ph ], [ %inc, %for.body8 ]
  %chosen.0101 = phi i32 [ 0, %for.body8.lr.ph ], [ %chosen.1, %for.body8 ]
  %conv10 = sext i32 %chosen.0101 to i64
  %add = add i64 %mul, %conv10
  %arrayidx11 = getelementptr inbounds float, float* %5, i64 %add
  %7 = load float, float* %arrayidx11, align 4, !tbaa !25
  %add15 = add i64 %conv5103, %mul
  %arrayidx16 = getelementptr inbounds float, float* %5, i64 %add15
  %8 = load float, float* %arrayidx16, align 4, !tbaa !25
  %cmp17 = fcmp olt float %7, %8
  %chosen.1 = select i1 %cmp17, i32 %id.0102, i32 %chosen.0101
  %inc = add i32 %id.0102, 1
  %conv5 = zext i32 %inc to i64
  %cmp6 = icmp ugt i64 %3, %conv5
  br i1 %cmp6, label %for.body8, label %for.cond.cleanup7

if.then37:                                        ; preds = %for.cond.cleanup
  %9 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to i8*
  call void @llvm.lifetime.start.p0i8(i64 376, i8* nonnull %9) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
  %10 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to %"class.std::basic_ostream"*
  %call38 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSt8ios_baseS0_E(%"class.std::basic_ostream"* nonnull %10, %"class.std::ios_base"* (%"class.std::ios_base"*)* nonnull @_ZSt5fixedRSt8ios_base)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %if.then37
  %call40 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call38, float %conv32)
          to label %invoke.cont39 unwind label %lpad

invoke.cont39:                                    ; preds = %invoke.cont
  %11 = bitcast %"class.std::__cxx11::basic_string"* %print_str to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %11) #20
  invoke void @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%"class.std::__cxx11::basic_string"* nonnull sret %print_str, %"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
          to label %invoke.cont42 unwind label %lpad41

invoke.cont42:                                    ; preds = %invoke.cont39
  %call43 = call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %call44 = call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %12 = call i64 @fwrite_unlocked(i8* %call43, i64 1, i64 %call44, %struct._IO_FILE* nonnull %call35)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %11) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %9) #20
  br label %if.end50

lpad:                                             ; preds = %invoke.cont, %if.then37
  %13 = landingpad { i8*, i32 }
          cleanup
  %14 = extractvalue { i8*, i32 } %13, 0
  %15 = extractvalue { i8*, i32 } %13, 1
  br label %ehcleanup48

lpad41:                                           ; preds = %invoke.cont39
  %16 = landingpad { i8*, i32 }
          cleanup
  %17 = extractvalue { i8*, i32 } %16, 1
  %18 = extractvalue { i8*, i32 } %16, 0
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %11) #20
  br label %ehcleanup48

ehcleanup48:                                      ; preds = %lpad41, %lpad
  %exn.slot.1 = phi i8* [ %18, %lpad41 ], [ %14, %lpad ]
  %ehselector.slot.1 = phi i32 [ %17, %lpad41 ], [ %15, %lpad ]
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %9) #20
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.1, 0
  %lpad.val59 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.1, 1
  resume { i8*, i32 } %lpad.val59

if.end50:                                         ; preds = %invoke.cont42, %for.cond.cleanup
  %call51 = call i32 @fclose(%struct._IO_FILE* %call35)
  ret float %conv32
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%"class.std::__cxx11::basic_ostringstream"* %this) unnamed_addr #0 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %0 = bitcast %"class.std::__cxx11::basic_ostringstream"* %this to i8*
  %1 = getelementptr inbounds %"class.std::__cxx11::basic_ostringstream", %"class.std::__cxx11::basic_ostringstream"* %this, i64 0, i32 2
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEEC2Ev(%"class.std::basic_ios"* nonnull %1)
  %2 = bitcast %"class.std::__cxx11::basic_ostringstream"* %this to %"class.std::basic_ostream"*
  invoke void @_ZNSoC2Ev(%"class.std::basic_ostream"* %2, i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, i64 0, i64 1))
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %3 = getelementptr inbounds %"class.std::__cxx11::basic_ostringstream", %"class.std::__cxx11::basic_ostringstream"* %this, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*], [5 x i8*] }, { [5 x i8*], [5 x i8*] }* @_ZTVNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, i64 0, inrange i32 0, i64 3) to i32 (...)**), i32 (...)*** %3, align 8, !tbaa !28
  %4 = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %1, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [5 x i8*], [5 x i8*] }, { [5 x i8*], [5 x i8*] }* @_ZTVNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, i64 0, inrange i32 1, i64 3) to i32 (...)**), i32 (...)*** %4, align 8, !tbaa !28
  %_M_stringbuf = getelementptr inbounds %"class.std::__cxx11::basic_ostringstream", %"class.std::__cxx11::basic_ostringstream"* %this, i64 0, i32 1
  invoke void @_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEC2ESt13_Ios_Openmode(%"class.std::__cxx11::basic_stringbuf"* nonnull %_M_stringbuf, i32 16)
          to label %invoke.cont3 unwind label %lpad2

invoke.cont3:                                     ; preds = %invoke.cont
  %5 = bitcast %"class.std::__cxx11::basic_ostringstream"* %this to i8**
  %vtable = load i8*, i8** %5, align 8, !tbaa !28
  %vbase.offset.ptr = getelementptr i8, i8* %vtable, i64 -24
  %6 = bitcast i8* %vbase.offset.ptr to i64*
  %vbase.offset = load i64, i64* %6, align 8
  %add.ptr4 = getelementptr inbounds i8, i8* %0, i64 %vbase.offset
  %7 = bitcast i8* %add.ptr4 to %"class.std::basic_ios"*
  %8 = getelementptr inbounds %"class.std::__cxx11::basic_stringbuf", %"class.std::__cxx11::basic_stringbuf"* %_M_stringbuf, i64 0, i32 0
  invoke void @_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E(%"class.std::basic_ios"* %7, %"class.std::basic_streambuf"* nonnull %8)
          to label %invoke.cont7 unwind label %lpad6

invoke.cont7:                                     ; preds = %invoke.cont3
  ret void

lpad:                                             ; preds = %entry
  %9 = landingpad { i8*, i32 }
          cleanup
  %10 = extractvalue { i8*, i32 } %9, 0
  %11 = extractvalue { i8*, i32 } %9, 1
  br label %ehcleanup8

lpad2:                                            ; preds = %invoke.cont
  %12 = landingpad { i8*, i32 }
          cleanup
  %13 = extractvalue { i8*, i32 } %12, 0
  %14 = extractvalue { i8*, i32 } %12, 1
  br label %ehcleanup

lpad6:                                            ; preds = %invoke.cont3
  %15 = landingpad { i8*, i32 }
          cleanup
  %16 = extractvalue { i8*, i32 } %15, 0
  %17 = extractvalue { i8*, i32 } %15, 1
  tail call void @_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_stringbuf"* nonnull %_M_stringbuf) #20
  br label %ehcleanup

ehcleanup:                                        ; preds = %lpad6, %lpad2
  %exn.slot.0 = phi i8* [ %16, %lpad6 ], [ %13, %lpad2 ]
  %ehselector.slot.0 = phi i32 [ %17, %lpad6 ], [ %14, %lpad2 ]
  tail call void @_ZNSoD2Ev(%"class.std::basic_ostream"* nonnull %2, i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, i64 0, i64 1)) #20
  br label %ehcleanup8

ehcleanup8:                                       ; preds = %ehcleanup, %lpad
  %exn.slot.1 = phi i8* [ %exn.slot.0, %ehcleanup ], [ %10, %lpad ]
  %ehselector.slot.1 = phi i32 [ %ehselector.slot.0, %ehcleanup ], [ %11, %lpad ]
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEED2Ev(%"class.std::basic_ios"* nonnull %1) #20
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.1, 0
  %lpad.val9 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.1, 1
  resume { i8*, i32 } %lpad.val9
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSt8ios_baseS0_E(%"class.std::basic_ostream"* %this, %"class.std::ios_base"* (%"class.std::ios_base"*)* %__pf) local_unnamed_addr #0 align 2 {
entry:
  %0 = bitcast %"class.std::basic_ostream"* %this to i8**
  %vtable = load i8*, i8** %0, align 8, !tbaa !28
  %vbase.offset.ptr = getelementptr i8, i8* %vtable, i64 -24
  %1 = bitcast i8* %vbase.offset.ptr to i64*
  %vbase.offset = load i64, i64* %1, align 8
  %2 = bitcast %"class.std::basic_ostream"* %this to i8*
  %add.ptr = getelementptr inbounds i8, i8* %2, i64 %vbase.offset
  %3 = bitcast i8* %add.ptr to %"class.std::ios_base"*
  %call = tail call dereferenceable(216) %"class.std::ios_base"* %__pf(%"class.std::ios_base"* dereferenceable(216) %3)
  ret %"class.std::basic_ostream"* %this
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local dereferenceable(216) %"class.std::ios_base"* @_ZSt5fixedRSt8ios_base(%"class.std::ios_base"* dereferenceable(216) %__base) #10 comdat {
entry:
  %call = tail call i32 @_ZNSt8ios_base4setfESt13_Ios_FmtflagsS0_(%"class.std::ios_base"* nonnull %__base, i32 4, i32 260)
  ret %"class.std::ios_base"* %__base
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* %this, float %__f) local_unnamed_addr #0 align 2 {
entry:
  %conv = fpext float %__f to double
  %call = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"* %this, double %conv)
  ret %"class.std::basic_ostream"* %call
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%"class.std::__cxx11::basic_string"* noalias sret %agg.result, %"class.std::__cxx11::basic_ostringstream"* %this) local_unnamed_addr #0 align 2 {
entry:
  %_M_stringbuf = getelementptr inbounds %"class.std::__cxx11::basic_ostringstream", %"class.std::__cxx11::basic_ostringstream"* %this, i64 0, i32 1
  tail call void @_ZNKSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE3strEv(%"class.std::__cxx11::basic_string"* sret %agg.result, %"class.std::__cxx11::basic_stringbuf"* nonnull %_M_stringbuf)
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %this)
  ret i8* %call
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 {
entry:
  %_M_string_length = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 1
  %0 = load i64, i64* %_M_string_length, align 8, !tbaa !30
  ret i64 %0
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* %this) unnamed_addr #4 align 2 {
entry:
  tail call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_ostringstream"* %this, i8** getelementptr inbounds ([4 x i8*], [4 x i8*]* @_ZTTNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEE, i64 0, i64 0)) #20
  %0 = getelementptr inbounds %"class.std::__cxx11::basic_ostringstream", %"class.std::__cxx11::basic_ostringstream"* %this, i64 0, i32 2
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEED2Ev(%"class.std::basic_ios"* nonnull %0) #20
  ret void
}

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local zeroext i1 @_Z16descendFloatComp9ClassProbS_(i64 %obj1.coerce, i64 %obj2.coerce) #11 {
entry:
  %obj1.sroa.0.0.extract.trunc = trunc i64 %obj1.coerce to i32
  %0 = bitcast i32 %obj1.sroa.0.0.extract.trunc to float
  %obj2.sroa.0.0.extract.trunc = trunc i64 %obj2.coerce to i32
  %1 = bitcast i32 %obj2.sroa.0.0.extract.trunc to float
  %cmp = fcmp ogt float %0, %1
  ret i1 %cmp
}

; Function Attrs: uwtable
define dso_local float @_Z19computeTop5AccuracyPhiPvj(i8* nocapture readonly %labels, i32 %num_labels, i8* nocapture readonly %result_ptr, i32 %num_classes) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %elem_probs = alloca %"class.std::vector.3", align 8
  %cProb = alloca %struct.ClassProb, align 4
  %ss = alloca %"class.std::__cxx11::basic_ostringstream", align 8
  %print_str = alloca %"class.std::__cxx11::basic_string", align 8
  %dim_sizes = getelementptr inbounds i8, i8* %result_ptr, i64 96
  %0 = bitcast i8* %dim_sizes to i64**
  %1 = load i64*, i64** %0, align 8, !tbaa !19
  %2 = load i64, i64* %1, align 8, !tbaa !20
  %arrayidx3 = getelementptr inbounds i64, i64* %1, i64 1
  %3 = load i64, i64* %arrayidx3, align 8, !tbaa !20
  %host_data = getelementptr inbounds i8, i8* %result_ptr, i64 48
  %4 = bitcast i8* %host_data to float**
  %5 = load float*, float** %4, align 8, !tbaa !22
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.23, i64 0, i64 0), i64 %2, i64 %3)
  %cmp127 = icmp eq i32 %num_labels, 0
  br i1 %cmp127, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %6 = bitcast %"class.std::vector.3"* %elem_probs to i8*
  %cmp5123 = icmp eq i32 %num_classes, 0
  %7 = bitcast %struct.ClassProb* %cProb to i8*
  %8 = bitcast %struct.ClassProb* %cProb to i32*
  %index = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %cProb, i64 0, i32 1
  %wide.trip.count136 = zext i32 %num_labels to i64
  %wide.trip.count = zext i32 %num_classes to i64
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.cond.cleanup20
  %phitmp = zext i32 %spec.select to i64
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %num_errors.0.lcssa = phi i64 [ 0, %entry ], [ %phitmp, %for.cond.cleanup.loopexit ]
  %sub = sub i64 %2, %num_errors.0.lcssa
  %conv41 = uitofp i64 %sub to double
  %conv43 = uitofp i64 %2 to double
  %div = fdiv double %conv41, %conv43
  %mul45 = fmul double %div, 1.000000e+02
  %conv46 = fptrunc double %mul45 to float
  %conv47 = fpext float %conv46 to double
  %call48 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.20, i64 0, i64 0), double %conv47)
  %call49 = call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.22, i64 0, i64 0))
  %cmp50 = icmp eq %struct._IO_FILE* %call49, null
  br i1 %cmp50, label %if.end68, label %if.then51

for.body:                                         ; preds = %for.cond.cleanup20, %for.body.lr.ph
  %indvars.iv134 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next135, %for.cond.cleanup20 ]
  %num_errors.0130 = phi i32 [ 0, %for.body.lr.ph ], [ %spec.select, %for.cond.cleanup20 ]
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %6) #20
  call void @_ZNSt6vectorI9ClassProbSaIS0_EEC2Ev(%"class.std::vector.3"* nonnull %elem_probs) #20
  br i1 %cmp5123, label %for.cond.cleanup6, label %for.body7.lr.ph

for.body7.lr.ph:                                  ; preds = %for.body
  %mul = mul i64 %3, %indvars.iv134
  br label %for.body7

for.cond.cleanup6:                                ; preds = %invoke.cont, %for.body
  %call10 = call %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE5beginEv(%"class.std::vector.3"* nonnull %elem_probs) #20
  %call12 = call %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE3endEv(%"class.std::vector.3"* nonnull %elem_probs) #20
  invoke void @_ZSt4sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEEPFbS2_S2_EEvT_SA_T0_(%struct.ClassProb* %call10, %struct.ClassProb* %call12, i1 (i64, i64)* nonnull @_Z16descendFloatComp9ClassProbS_)
          to label %for.cond18.preheader unwind label %lpad16

for.cond18.preheader:                             ; preds = %for.cond.cleanup6
  %arrayidx26 = getelementptr inbounds i8, i8* %labels, i64 %indvars.iv134
  br label %for.body21

for.body7:                                        ; preds = %invoke.cont, %for.body7.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body7.lr.ph ], [ %indvars.iv.next, %invoke.cont ]
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %7) #20
  %add = add i64 %mul, %indvars.iv
  %arrayidx9 = getelementptr inbounds float, float* %5, i64 %add
  %9 = bitcast float* %arrayidx9 to i32*
  %10 = load i32, i32* %9, align 4, !tbaa !25
  store i32 %10, i32* %8, align 4, !tbaa !33
  %11 = trunc i64 %indvars.iv to i32
  store i32 %11, i32* %index, align 4, !tbaa !35
  invoke void @_ZNSt6vectorI9ClassProbSaIS0_EE9push_backERKS0_(%"class.std::vector.3"* nonnull %elem_probs, %struct.ClassProb* nonnull dereferenceable(8) %cProb)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %for.body7
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %7) #20
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup6, label %for.body7

lpad:                                             ; preds = %for.body7
  %12 = landingpad { i8*, i32 }
          cleanup
  %13 = bitcast %struct.ClassProb* %cProb to i8*
  %14 = extractvalue { i8*, i32 } %12, 0
  %15 = extractvalue { i8*, i32 } %12, 1
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %13) #20
  br label %ehcleanup

for.cond.cleanup20:                               ; preds = %for.body21
  %16 = and i8 %spec.select118, 1
  %17 = xor i8 %16, 1
  %18 = zext i8 %17 to i32
  %spec.select = add nuw nsw i32 %num_errors.0130, %18
  call void @_ZNSt6vectorI9ClassProbSaIS0_EED2Ev(%"class.std::vector.3"* nonnull %elem_probs) #20
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %6) #20
  %indvars.iv.next135 = add nuw nsw i64 %indvars.iv134, 1
  %exitcond137 = icmp eq i64 %indvars.iv.next135, %wide.trip.count136
  br i1 %exitcond137, label %for.cond.cleanup.loopexit, label %for.body

lpad16:                                           ; preds = %for.cond.cleanup6
  %19 = landingpad { i8*, i32 }
          cleanup
  %20 = extractvalue { i8*, i32 } %19, 0
  %21 = extractvalue { i8*, i32 } %19, 1
  br label %ehcleanup

for.body21:                                       ; preds = %for.body21, %for.cond18.preheader
  %indvars.iv131 = phi i64 [ 0, %for.cond18.preheader ], [ %indvars.iv.next132, %for.body21 ]
  %matched.0125 = phi i8 [ 0, %for.cond18.preheader ], [ %spec.select118, %for.body21 ]
  %call24 = call dereferenceable(8) %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EEixEm(%"class.std::vector.3"* nonnull %elem_probs, i64 %indvars.iv131) #20
  %cProb22.sroa.3.0..sroa_idx87 = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %call24, i64 0, i32 1
  %cProb22.sroa.3.0.copyload = load i32, i32* %cProb22.sroa.3.0..sroa_idx87, align 4
  %22 = load i8, i8* %arrayidx26, align 1, !tbaa !36
  %conv27 = zext i8 %22 to i32
  %cmp28 = icmp eq i32 %cProb22.sroa.3.0.copyload, %conv27
  %spec.select118 = select i1 %cmp28, i8 1, i8 %matched.0125
  %indvars.iv.next132 = add nuw nsw i64 %indvars.iv131, 1
  %exitcond133 = icmp eq i64 %indvars.iv.next132, 5
  br i1 %exitcond133, label %for.cond.cleanup20, label %for.body21

ehcleanup:                                        ; preds = %lpad16, %lpad
  %ehselector.slot.0 = phi i32 [ %15, %lpad ], [ %21, %lpad16 ]
  %exn.slot.0 = phi i8* [ %14, %lpad ], [ %20, %lpad16 ]
  %23 = bitcast %"class.std::vector.3"* %elem_probs to i8*
  call void @_ZNSt6vectorI9ClassProbSaIS0_EED2Ev(%"class.std::vector.3"* nonnull %elem_probs) #20
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %23) #20
  br label %ehcleanup72

if.then51:                                        ; preds = %for.cond.cleanup
  %24 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to i8*
  call void @llvm.lifetime.start.p0i8(i64 376, i8* nonnull %24) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
  %25 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to %"class.std::basic_ostream"*
  %call54 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSt8ios_baseS0_E(%"class.std::basic_ostream"* nonnull %25, %"class.std::ios_base"* (%"class.std::ios_base"*)* nonnull @_ZSt5fixedRSt8ios_base)
          to label %invoke.cont53 unwind label %lpad52

invoke.cont53:                                    ; preds = %if.then51
  %call56 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call54, float %conv46)
          to label %invoke.cont55 unwind label %lpad52

invoke.cont55:                                    ; preds = %invoke.cont53
  %26 = bitcast %"class.std::__cxx11::basic_string"* %print_str to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %26) #20
  invoke void @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%"class.std::__cxx11::basic_string"* nonnull sret %print_str, %"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
          to label %invoke.cont58 unwind label %lpad57

invoke.cont58:                                    ; preds = %invoke.cont55
  %call59 = call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %call60 = call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %27 = call i64 @fwrite_unlocked(i8* %call59, i64 1, i64 %call60, %struct._IO_FILE* nonnull %call49)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %26) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %24) #20
  br label %if.end68

lpad52:                                           ; preds = %invoke.cont53, %if.then51
  %28 = landingpad { i8*, i32 }
          cleanup
  %29 = extractvalue { i8*, i32 } %28, 0
  %30 = extractvalue { i8*, i32 } %28, 1
  br label %ehcleanup66

lpad57:                                           ; preds = %invoke.cont55
  %31 = landingpad { i8*, i32 }
          cleanup
  %32 = extractvalue { i8*, i32 } %31, 1
  %33 = extractvalue { i8*, i32 } %31, 0
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %26) #20
  br label %ehcleanup66

ehcleanup66:                                      ; preds = %lpad57, %lpad52
  %ehselector.slot.2 = phi i32 [ %32, %lpad57 ], [ %30, %lpad52 ]
  %exn.slot.2 = phi i8* [ %33, %lpad57 ], [ %29, %lpad52 ]
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %24) #20
  br label %ehcleanup72

if.end68:                                         ; preds = %invoke.cont58, %for.cond.cleanup
  %call69 = call i32 @fclose(%struct._IO_FILE* %call49)
  ret float %conv46

ehcleanup72:                                      ; preds = %ehcleanup66, %ehcleanup
  %ehselector.slot.3 = phi i32 [ %ehselector.slot.0, %ehcleanup ], [ %ehselector.slot.2, %ehcleanup66 ]
  %exn.slot.3 = phi i8* [ %exn.slot.0, %ehcleanup ], [ %exn.slot.2, %ehcleanup66 ]
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.3, 0
  %lpad.val77 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.3, 1
  resume { i8*, i32 } %lpad.val77
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6vectorI9ClassProbSaIS0_EEC2Ev(%"class.std::vector.3"* %this) unnamed_addr #4 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0
  tail call void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EEC2Ev(%"struct.std::_Vector_base.4"* %0) #20
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt6vectorI9ClassProbSaIS0_EE9push_backERKS0_(%"class.std::vector.3"* %this, %struct.ClassProb* dereferenceable(8) %__x) local_unnamed_addr #0 comdat align 2 {
entry:
  %_M_finish = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 1
  %0 = load %struct.ClassProb*, %struct.ClassProb** %_M_finish, align 8, !tbaa !37
  %_M_end_of_storage = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 2
  %1 = load %struct.ClassProb*, %struct.ClassProb** %_M_end_of_storage, align 8, !tbaa !39
  %cmp = icmp eq %struct.ClassProb* %0, %1
  br i1 %cmp, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %2 = bitcast %"class.std::vector.3"* %this to %"class.std::allocator.5"*
  tail call void @_ZNSt16allocator_traitsISaI9ClassProbEE9constructIS0_JRKS0_EEEvRS1_PT_DpOT0_(%"class.std::allocator.5"* dereferenceable(1) %2, %struct.ClassProb* %0, %struct.ClassProb* nonnull dereferenceable(8) %__x) #20
  %3 = load %struct.ClassProb*, %struct.ClassProb** %_M_finish, align 8, !tbaa !37
  %incdec.ptr = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %3, i64 1
  store %struct.ClassProb* %incdec.ptr, %struct.ClassProb** %_M_finish, align 8, !tbaa !37
  br label %if.end

if.else:                                          ; preds = %entry
  %call = tail call %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE3endEv(%"class.std::vector.3"* nonnull %this) #20
  tail call void @_ZNSt6vectorI9ClassProbSaIS0_EE17_M_realloc_insertIJRKS0_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_(%"class.std::vector.3"* nonnull %this, %struct.ClassProb* %call, %struct.ClassProb* nonnull dereferenceable(8) %__x)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZSt4sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEEPFbS2_S2_EEvT_SA_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, i1 (i64, i64)* %__comp) local_unnamed_addr #10 comdat {
entry:
  %call = tail call i1 (i64, i64)* @_ZN9__gnu_cxx5__ops16__iter_comp_iterIPFb9ClassProbS2_EEENS0_15_Iter_comp_iterIT_EES6_(i1 (i64, i64)* %__comp)
  tail call void @_ZSt6__sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, i1 (i64, i64)* %call)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE5beginEv(%"class.std::vector.3"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %retval = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %_M_start = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 0
  call void @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEC2ERKS2_(%"class.__gnu_cxx::__normal_iterator"* nonnull %retval, %struct.ClassProb** dereferenceable(8) %_M_start) #20
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %retval, i64 0, i32 0
  %0 = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  ret %struct.ClassProb* %0
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE3endEv(%"class.std::vector.3"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %retval = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %_M_finish = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 1
  call void @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEC2ERKS2_(%"class.__gnu_cxx::__normal_iterator"* nonnull %retval, %struct.ClassProb** nonnull dereferenceable(8) %_M_finish) #20
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %retval, i64 0, i32 0
  %0 = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  ret %struct.ClassProb* %0
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EEixEm(%"class.std::vector.3"* %this, i64 %__n) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_start = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 0
  %0 = load %struct.ClassProb*, %struct.ClassProb** %_M_start, align 8, !tbaa !40
  %add.ptr = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %0, i64 %__n
  ret %struct.ClassProb* %add.ptr
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1 immarg) #5

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt6vectorI9ClassProbSaIS0_EED2Ev(%"class.std::vector.3"* %this) unnamed_addr #4 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %0 = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0
  %_M_start = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 0
  %1 = load %struct.ClassProb*, %struct.ClassProb** %_M_start, align 8, !tbaa !40
  %_M_finish = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 1
  %2 = load %struct.ClassProb*, %struct.ClassProb** %_M_finish, align 8, !tbaa !37
  %call = tail call dereferenceable(1) %"class.std::allocator.5"* @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base.4"* %0) #20
  invoke void @_ZSt8_DestroyIP9ClassProbS0_EvT_S2_RSaIT0_E(%struct.ClassProb* %1, %struct.ClassProb* %2, %"class.std::allocator.5"* nonnull dereferenceable(1) %call)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  tail call void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EED2Ev(%"struct.std::_Vector_base.4"* %0) #20
  ret void

lpad:                                             ; preds = %entry
  %3 = landingpad { i8*, i32 }
          catch i8* null
  %4 = extractvalue { i8*, i32 } %3, 0
  tail call void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EED2Ev(%"struct.std::_Vector_base.4"* %0) #20
  tail call void @__clang_call_terminate(i8* %4) #21
  unreachable
}

; Function Attrs: uwtable
define dso_local void @_Z17dumpFinalAccuracyf(float %accuracy) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %accuracy.addr = alloca float, align 4
  %ss = alloca %"class.std::__cxx11::basic_ostringstream", align 8
  %print_str = alloca %"class.std::__cxx11::basic_string", align 8
  store float %accuracy, float* %accuracy.addr, align 4, !tbaa !25
  %conv = fpext float %accuracy to double
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.24, i64 0, i64 0), double %conv)
  %call1 = tail call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.22, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call1, null
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to i8*
  call void @llvm.lifetime.start.p0i8(i64 376, i8* nonnull %0) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
  %1 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to %"class.std::basic_ostream"*
  %call2 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSt8ios_baseS0_E(%"class.std::basic_ostream"* nonnull %1, %"class.std::ios_base"* (%"class.std::ios_base"*)* nonnull @_ZSt5fixedRSt8ios_base)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %if.then
  %2 = load float, float* %accuracy.addr, align 4, !tbaa !25
  %call4 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call2, float %2)
          to label %invoke.cont3 unwind label %lpad

invoke.cont3:                                     ; preds = %invoke.cont
  %3 = bitcast %"class.std::__cxx11::basic_string"* %print_str to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %3) #20
  invoke void @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%"class.std::__cxx11::basic_string"* nonnull sret %print_str, %"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
          to label %invoke.cont6 unwind label %lpad5

invoke.cont6:                                     ; preds = %invoke.cont3
  %call7 = call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %call8 = call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %4 = call i64 @fwrite_unlocked(i8* %call7, i64 1, i64 %call8, %struct._IO_FILE* nonnull %call1)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %3) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %0) #20
  br label %if.end

lpad:                                             ; preds = %invoke.cont, %if.then
  %5 = landingpad { i8*, i32 }
          cleanup
  %6 = extractvalue { i8*, i32 } %5, 0
  %7 = extractvalue { i8*, i32 } %5, 1
  br label %ehcleanup12

lpad5:                                            ; preds = %invoke.cont3
  %8 = landingpad { i8*, i32 }
          cleanup
  %9 = extractvalue { i8*, i32 } %8, 1
  %10 = extractvalue { i8*, i32 } %8, 0
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %3) #20
  br label %ehcleanup12

ehcleanup12:                                      ; preds = %lpad5, %lpad
  %exn.slot.1 = phi i8* [ %10, %lpad5 ], [ %6, %lpad ]
  %ehselector.slot.1 = phi i32 [ %9, %lpad5 ], [ %7, %lpad ]
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %0) #20
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.1, 0
  %lpad.val16 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.1, 1
  resume { i8*, i32 } %lpad.val16

if.end:                                           ; preds = %invoke.cont6, %entry
  %call14 = call i32 @fclose(%struct._IO_FILE* %call1)
  call void @_ZNSt6vectorIfSaIfEE9push_backERKf(%"class.std::vector"* nonnull @run_accuracies, float* nonnull dereferenceable(4) %accuracy.addr)
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt6vectorIfSaIfEE9push_backERKf(%"class.std::vector"* %this, float* dereferenceable(4) %__x) local_unnamed_addr #0 comdat align 2 {
entry:
  %_M_finish = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 1
  %0 = load float*, float** %_M_finish, align 8, !tbaa !11
  %_M_end_of_storage = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 2
  %1 = load float*, float** %_M_end_of_storage, align 8, !tbaa !41
  %cmp = icmp eq float* %0, %1
  br i1 %cmp, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %2 = bitcast %"class.std::vector"* %this to %"class.std::allocator"*
  tail call void @_ZNSt16allocator_traitsISaIfEE9constructIfJRKfEEEvRS0_PT_DpOT0_(%"class.std::allocator"* dereferenceable(1) %2, float* %0, float* nonnull dereferenceable(4) %__x) #20
  %3 = load float*, float** %_M_finish, align 8, !tbaa !11
  %incdec.ptr = getelementptr inbounds float, float* %3, i64 1
  store float* %incdec.ptr, float** %_M_finish, align 8, !tbaa !11
  br label %if.end

if.else:                                          ; preds = %entry
  %call = tail call float* @_ZNSt6vectorIfSaIfEE3endEv(%"class.std::vector"* nonnull %this) #20
  tail call void @_ZNSt6vectorIfSaIfEE17_M_realloc_insertIJRKfEEEvN9__gnu_cxx17__normal_iteratorIPfS1_EEDpOT_(%"class.std::vector"* nonnull %this, float* %call, float* nonnull dereferenceable(4) %__x)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; Function Attrs: uwtable
define dso_local void @_Z11dumpAvgPSNRf(float %avg_psnr) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %ss = alloca %"class.std::__cxx11::basic_ostringstream", align 8
  %print_str = alloca %"class.std::__cxx11::basic_string", align 8
  %call = tail call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.25, i64 0, i64 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.22, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call, null
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to i8*
  call void @llvm.lifetime.start.p0i8(i64 376, i8* nonnull %0) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
  %1 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to %"class.std::basic_ostream"*
  %call1 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSt8ios_baseS0_E(%"class.std::basic_ostream"* nonnull %1, %"class.std::ios_base"* (%"class.std::ios_base"*)* nonnull @_ZSt5fixedRSt8ios_base)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %if.then
  %call3 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call1, float %avg_psnr)
          to label %invoke.cont2 unwind label %lpad

invoke.cont2:                                     ; preds = %invoke.cont
  %2 = bitcast %"class.std::__cxx11::basic_string"* %print_str to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %2) #20
  invoke void @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%"class.std::__cxx11::basic_string"* nonnull sret %print_str, %"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
          to label %invoke.cont5 unwind label %lpad4

invoke.cont5:                                     ; preds = %invoke.cont2
  %call6 = call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %call7 = call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %3 = call i64 @fwrite_unlocked(i8* %call6, i64 1, i64 %call7, %struct._IO_FILE* nonnull %call)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %2) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %0) #20
  br label %if.end

lpad:                                             ; preds = %invoke.cont, %if.then
  %4 = landingpad { i8*, i32 }
          cleanup
  %5 = extractvalue { i8*, i32 } %4, 0
  %6 = extractvalue { i8*, i32 } %4, 1
  br label %ehcleanup11

lpad4:                                            ; preds = %invoke.cont2
  %7 = landingpad { i8*, i32 }
          cleanup
  %8 = extractvalue { i8*, i32 } %7, 1
  %9 = extractvalue { i8*, i32 } %7, 0
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %2) #20
  br label %ehcleanup11

ehcleanup11:                                      ; preds = %lpad4, %lpad
  %exn.slot.1 = phi i8* [ %9, %lpad4 ], [ %5, %lpad ]
  %ehselector.slot.1 = phi i32 [ %8, %lpad4 ], [ %6, %lpad ]
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %0) #20
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.1, 0
  %lpad.val15 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.1, 1
  resume { i8*, i32 } %lpad.val15

if.end:                                           ; preds = %invoke.cont5, %entry
  %call13 = call i32 @fclose(%struct._IO_FILE* %call)
  ret void
}

; Function Attrs: uwtable
define dso_local void @_Z11dumpPSNRStdf(float %psnr_std) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %ss = alloca %"class.std::__cxx11::basic_ostringstream", align 8
  %print_str = alloca %"class.std::__cxx11::basic_string", align 8
  %call = tail call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.26, i64 0, i64 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.22, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call, null
  br i1 %cmp, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to i8*
  call void @llvm.lifetime.start.p0i8(i64 376, i8* nonnull %0) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
  %1 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to %"class.std::basic_ostream"*
  %call1 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSt8ios_baseS0_E(%"class.std::basic_ostream"* nonnull %1, %"class.std::ios_base"* (%"class.std::ios_base"*)* nonnull @_ZSt5fixedRSt8ios_base)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %if.then
  %call3 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call1, float %psnr_std)
          to label %invoke.cont2 unwind label %lpad

invoke.cont2:                                     ; preds = %invoke.cont
  %2 = bitcast %"class.std::__cxx11::basic_string"* %print_str to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %2) #20
  invoke void @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%"class.std::__cxx11::basic_string"* nonnull sret %print_str, %"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
          to label %invoke.cont5 unwind label %lpad4

invoke.cont5:                                     ; preds = %invoke.cont2
  %call6 = call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %call7 = call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %3 = call i64 @fwrite_unlocked(i8* %call6, i64 1, i64 %call7, %struct._IO_FILE* nonnull %call)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %2) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %0) #20
  br label %if.end

lpad:                                             ; preds = %invoke.cont, %if.then
  %4 = landingpad { i8*, i32 }
          cleanup
  %5 = extractvalue { i8*, i32 } %4, 0
  %6 = extractvalue { i8*, i32 } %4, 1
  br label %ehcleanup11

lpad4:                                            ; preds = %invoke.cont2
  %7 = landingpad { i8*, i32 }
          cleanup
  %8 = extractvalue { i8*, i32 } %7, 1
  %9 = extractvalue { i8*, i32 } %7, 0
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %2) #20
  br label %ehcleanup11

ehcleanup11:                                      ; preds = %lpad4, %lpad
  %exn.slot.1 = phi i8* [ %9, %lpad4 ], [ %5, %lpad ]
  %ehselector.slot.1 = phi i32 [ %8, %lpad4 ], [ %6, %lpad ]
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %0) #20
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.1, 0
  %lpad.val15 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.1, 1
  resume { i8*, i32 } %lpad.val15

if.end:                                           ; preds = %invoke.cont5, %entry
  %call13 = call i32 @fclose(%struct._IO_FILE* %call)
  ret void
}

; Function Attrs: uwtable
define dso_local void @_Z23dumpExecutionAccuraciesv() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %ss = alloca %"class.std::__cxx11::basic_ostringstream", align 8
  %print_str = alloca %"class.std::__cxx11::basic_string", align 8
  %call = tail call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.27, i64 0, i64 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.22, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call, null
  br i1 %cmp, label %if.end, label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry
  %call137 = call i64 @_ZNKSt6vectorIfSaIfEE4sizeEv(%"class.std::vector"* nonnull @run_accuracies) #20
  %cmp238 = icmp eq i64 %call137, 0
  br i1 %cmp238, label %if.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  %0 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to i8*
  %1 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to %"class.std::basic_ostream"*
  %2 = bitcast %"class.std::__cxx11::basic_string"* %print_str to i8*
  br label %for.body

for.body:                                         ; preds = %invoke.cont9, %for.body.lr.ph
  %conv40 = phi i64 [ 0, %for.body.lr.ph ], [ %conv, %invoke.cont9 ]
  %i.039 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %invoke.cont9 ]
  %call4 = call dereferenceable(4) float* @_ZNSt6vectorIfSaIfEEixEm(%"class.std::vector"* nonnull @run_accuracies, i64 %conv40) #20
  %3 = load float, float* %call4, align 4, !tbaa !25
  call void @llvm.lifetime.start.p0i8(i64 376, i8* nonnull %0) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
  %call5 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSt8ios_baseS0_E(%"class.std::basic_ostream"* nonnull %1, %"class.std::ios_base"* (%"class.std::ios_base"*)* nonnull @_ZSt5fixedRSt8ios_base)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %for.body
  %call7 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call5, float %3)
          to label %invoke.cont6 unwind label %lpad

invoke.cont6:                                     ; preds = %invoke.cont
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %2) #20
  invoke void @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%"class.std::__cxx11::basic_string"* nonnull sret %print_str, %"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
          to label %invoke.cont9 unwind label %lpad8

invoke.cont9:                                     ; preds = %invoke.cont6
  %call10 = call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %call11 = call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %4 = call i64 @fwrite_unlocked(i8* %call10, i64 1, i64 %call11, %struct._IO_FILE* nonnull %call)
  %fputc_unlocked = call i32 @fputc_unlocked(i32 10, %struct._IO_FILE* nonnull %call)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %2) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %0) #20
  %inc = add i32 %i.039, 1
  %conv = zext i32 %inc to i64
  %call1 = call i64 @_ZNKSt6vectorIfSaIfEE4sizeEv(%"class.std::vector"* nonnull @run_accuracies) #20
  %cmp2 = icmp ugt i64 %call1, %conv
  br i1 %cmp2, label %for.body, label %if.end

lpad:                                             ; preds = %invoke.cont, %for.body
  %5 = landingpad { i8*, i32 }
          cleanup
  %6 = extractvalue { i8*, i32 } %5, 0
  %7 = extractvalue { i8*, i32 } %5, 1
  br label %ehcleanup17

lpad8:                                            ; preds = %invoke.cont6
  %8 = landingpad { i8*, i32 }
          cleanup
  %9 = bitcast %"class.std::__cxx11::basic_string"* %print_str to i8*
  %10 = extractvalue { i8*, i32 } %8, 1
  %11 = extractvalue { i8*, i32 } %8, 0
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %9) #20
  br label %ehcleanup17

ehcleanup17:                                      ; preds = %lpad8, %lpad
  %exn.slot.1 = phi i8* [ %11, %lpad8 ], [ %6, %lpad ]
  %ehselector.slot.1 = phi i32 [ %10, %lpad8 ], [ %7, %lpad ]
  %12 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to i8*
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %12) #20
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.1, 0
  %lpad.val23 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.1, 1
  resume { i8*, i32 } %lpad.val23

if.end:                                           ; preds = %invoke.cont9, %for.cond.preheader, %entry
  %call21 = call i32 @fclose(%struct._IO_FILE* %call)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNKSt6vectorIfSaIfEE4sizeEv(%"class.std::vector"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_finish = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 1
  %0 = bitcast float** %_M_finish to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !11
  %2 = bitcast %"class.std::vector"* %this to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !6
  %sub.ptr.sub = sub i64 %1, %3
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 2
  ret i64 %sub.ptr.div
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(4) float* @_ZNSt6vectorIfSaIfEEixEm(%"class.std::vector"* %this, i64 %__n) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_start = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 0
  %0 = load float*, float** %_M_start, align 8, !tbaa !6
  %add.ptr = getelementptr inbounds float, float* %0, i64 %__n
  ret float* %add.ptr
}

; Function Attrs: uwtable
define dso_local float @_Z16readPSNRFromFilePKc(i8* nocapture readonly %file_name) local_unnamed_addr #0 {
entry:
  %psnr = alloca float, align 4
  %0 = bitcast float* %psnr to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #20
  %call = tail call %struct._IO_FILE* @fopen(i8* %file_name, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.28, i64 0, i64 0))
  %cmp = icmp eq %struct._IO_FILE* %call, null
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @str.50, i64 0, i64 0))
  tail call void @abort() #21
  unreachable

if.end:                                           ; preds = %entry
  %call2 = call i32 (%struct._IO_FILE*, i8*, ...) @__isoc99_fscanf(%struct._IO_FILE* nonnull %call, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.30, i64 0, i64 0), float* nonnull %psnr)
  %1 = load float, float* %psnr, align 4, !tbaa !25
  %conv = fpext float %1 to double
  %call3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.31, i64 0, i64 0), double %conv)
  %2 = load float, float* %psnr, align 4, !tbaa !25
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #20
  ret float %2
}

declare dso_local i32 @__isoc99_fscanf(%struct._IO_FILE*, i8*, ...) local_unnamed_addr #1

; Function Attrs: uwtable
define dso_local float @_Z20computePSNRViolationPvS_f(i8* nocapture readonly %gold_ptr, i8* nocapture readonly %approx_ptr, float %PSNR_threshold) local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %psnr_list = alloca %"class.std::vector", align 8
  %psnr = alloca float, align 4
  %ss = alloca %"class.std::__cxx11::basic_ostringstream", align 8
  %print_str = alloca %"class.std::__cxx11::basic_string", align 8
  %call = tail call float @_Z16readPSNRFromFilePKc(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.32, i64 0, i64 0))
  %0 = bitcast %"class.std::vector"* %psnr_list to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %0) #20
  call void @_ZNSt6vectorIfSaIfEEC2Ev(%"class.std::vector"* nonnull %psnr_list) #20
  %dim_sizes1 = getelementptr inbounds i8, i8* %gold_ptr, i64 96
  %1 = bitcast i8* %dim_sizes1 to i64**
  %2 = load i64*, i64** %1, align 8, !tbaa !19
  %3 = load i64, i64* %2, align 8, !tbaa !20
  %arrayidx2 = getelementptr inbounds i64, i64* %2, i64 1
  %4 = load i64, i64* %arrayidx2, align 8, !tbaa !20
  %arrayidx3 = getelementptr inbounds i64, i64* %2, i64 2
  %5 = load i64, i64* %arrayidx3, align 8, !tbaa !20
  %mul = mul i64 %5, %4
  %arrayidx4 = getelementptr inbounds i64, i64* %2, i64 3
  %6 = load i64, i64* %arrayidx4, align 8, !tbaa !20
  %mul5 = mul i64 %mul, %6
  %call6 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([36 x i8], [36 x i8]* @.str.33, i64 0, i64 0), i64 %3, i64 %mul5)
  %host_data = getelementptr inbounds i8, i8* %gold_ptr, i64 48
  %7 = bitcast i8* %host_data to float**
  %8 = load float*, float** %7, align 8, !tbaa !22
  %host_data7 = getelementptr inbounds i8, i8* %approx_ptr, i64 48
  %9 = bitcast i8* %host_data7 to float**
  %10 = load float*, float** %9, align 8, !tbaa !22
  %call10 = call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.34, i64 0, i64 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.22, i64 0, i64 0))
  %cmp237 = icmp eq i64 %3, 0
  br i1 %cmp237, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %cmp13232 = icmp eq i64 %mul5, 0
  %conv = uitofp i64 %mul5 to float
  %11 = bitcast float* %psnr to i8*
  %12 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to i8*
  %13 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to %"class.std::basic_ostream"*
  %14 = bitcast %"class.std::__cxx11::basic_string"* %print_str to i8*
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %invoke.cont50
  %phitmp = sitofp i32 %num_errors.1 to double
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %num_errors.0.lcssa = phi double [ 0.000000e+00, %entry ], [ %phitmp, %for.cond.cleanup.loopexit ]
  %sum_psnr.0.lcssa = phi float [ 0.000000e+00, %entry ], [ %add33, %for.cond.cleanup.loopexit ]
  %conv70 = uitofp i64 %3 to double
  %div71 = fdiv double %num_errors.0.lcssa, %conv70
  %mul72 = fmul double %div71, 1.000000e+02
  %conv73 = fptrunc double %mul72 to float
  %conv74 = fpext float %conv73 to double
  %call77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.36, i64 0, i64 0), double %conv74)
  %conv78 = uitofp i64 %3 to float
  %div79 = fdiv float %sum_psnr.0.lcssa, %conv78
  %conv80 = fpext float %div79 to double
  %call83 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.37, i64 0, i64 0), double %conv80)
  invoke void @_Z11dumpAvgPSNRf(float %div79)
          to label %invoke.cont84 unwind label %lpad81

for.body:                                         ; preds = %invoke.cont50, %for.body.lr.ph
  %sum_psnr.0240 = phi float [ 0.000000e+00, %for.body.lr.ph ], [ %add33, %invoke.cont50 ]
  %num_errors.0239 = phi i32 [ 0, %for.body.lr.ph ], [ %num_errors.1, %invoke.cont50 ]
  %i.0238 = phi i64 [ 0, %for.body.lr.ph ], [ %inc65, %invoke.cont50 ]
  %mul11 = mul i64 %i.0238, %mul5
  br i1 %cmp13232, label %for.cond.cleanup14, label %for.body15

for.cond.cleanup14:                               ; preds = %for.body15, %for.body
  %mse_sum.0.lcssa = phi float [ 0.000000e+00, %for.body ], [ %add20, %for.body15 ]
  %div = fdiv float %mse_sum.0.lcssa, %conv
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %11) #20
  %call28 = call float @_ZSt4sqrtf(float %div)
  %div29 = fdiv float 2.550000e+02, %call28
  %call31 = call float @_ZSt5log10f(float %div29)
  %mul32 = fmul float %call31, 2.000000e+01
  store float %mul32, float* %psnr, align 4, !tbaa !25
  %add33 = fadd float %sum_psnr.0240, %mul32
  %cmp34 = fcmp olt float %mul32, %call
  %add36 = zext i1 %cmp34 to i32
  %num_errors.1 = add nuw nsw i32 %num_errors.0239, %add36
  %conv38 = fpext float %mul32 to double
  %call40 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.35, i64 0, i64 0), double %conv38)
  invoke void @_ZNSt6vectorIfSaIfEE9push_backERKf(%"class.std::vector"* nonnull %psnr_list, float* nonnull dereferenceable(4) %psnr)
          to label %invoke.cont41 unwind label %lpad26

for.body15:                                       ; preds = %for.body15, %for.body
  %j.0235 = phi i64 [ %inc, %for.body15 ], [ 0, %for.body ]
  %mse_sum.0233 = phi float [ %add20, %for.body15 ], [ 0.000000e+00, %for.body ]
  %add = add i64 %j.0235, %mul11
  %arrayidx16 = getelementptr inbounds float, float* %8, i64 %add
  %15 = load float, float* %arrayidx16, align 4, !tbaa !25
  %arrayidx18 = getelementptr inbounds float, float* %10, i64 %add
  %16 = load float, float* %arrayidx18, align 4, !tbaa !25
  %sub = fsub float %15, %16
  %mul19 = fmul float %sub, %sub
  %add20 = fadd float %mse_sum.0233, %mul19
  %inc = add nuw i64 %j.0235, 1
  %exitcond251 = icmp eq i64 %inc, %mul5
  br i1 %exitcond251, label %for.cond.cleanup14, label %for.body15

lpad26:                                           ; preds = %for.cond.cleanup14
  %17 = landingpad { i8*, i32 }
          cleanup
  %18 = extractvalue { i8*, i32 } %17, 0
  %19 = extractvalue { i8*, i32 } %17, 1
  br label %ehcleanup60

invoke.cont41:                                    ; preds = %for.cond.cleanup14
  call void @llvm.lifetime.start.p0i8(i64 376, i8* nonnull %12) #20
  invoke void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEEC1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
          to label %invoke.cont43 unwind label %lpad42

invoke.cont43:                                    ; preds = %invoke.cont41
  %call46 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSt8ios_baseS0_E(%"class.std::basic_ostream"* nonnull %13, %"class.std::ios_base"* (%"class.std::ios_base"*)* nonnull @_ZSt5fixedRSt8ios_base)
          to label %invoke.cont45 unwind label %lpad44

invoke.cont45:                                    ; preds = %invoke.cont43
  %20 = load float, float* %psnr, align 4, !tbaa !25
  %call48 = invoke dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call46, float %20)
          to label %invoke.cont47 unwind label %lpad44

invoke.cont47:                                    ; preds = %invoke.cont45
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #20
  invoke void @_ZNKSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEE3strEv(%"class.std::__cxx11::basic_string"* nonnull sret %print_str, %"class.std::__cxx11::basic_ostringstream"* nonnull %ss)
          to label %invoke.cont50 unwind label %lpad49

invoke.cont50:                                    ; preds = %invoke.cont47
  %call51 = call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5c_strEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %call52 = call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6lengthEv(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  %21 = call i64 @fwrite_unlocked(i8* %call51, i64 1, i64 %call52, %struct._IO_FILE* %call10)
  %fputc_unlocked = call i32 @fputc_unlocked(i32 10, %struct._IO_FILE* %call10)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* nonnull %print_str) #20
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #20
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %12) #20
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %11) #20
  %inc65 = add nuw i64 %i.0238, 1
  %exitcond252 = icmp eq i64 %inc65, %3
  br i1 %exitcond252, label %for.cond.cleanup.loopexit, label %for.body

lpad42:                                           ; preds = %invoke.cont41
  %22 = landingpad { i8*, i32 }
          cleanup
  %23 = extractvalue { i8*, i32 } %22, 0
  %24 = extractvalue { i8*, i32 } %22, 1
  br label %ehcleanup59

lpad44:                                           ; preds = %invoke.cont45, %invoke.cont43
  %25 = landingpad { i8*, i32 }
          cleanup
  %26 = extractvalue { i8*, i32 } %25, 0
  %27 = extractvalue { i8*, i32 } %25, 1
  br label %ehcleanup58

lpad49:                                           ; preds = %invoke.cont47
  %28 = landingpad { i8*, i32 }
          cleanup
  %29 = bitcast %"class.std::__cxx11::basic_string"* %print_str to i8*
  %30 = extractvalue { i8*, i32 } %28, 1
  %31 = extractvalue { i8*, i32 } %28, 0
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %29) #20
  br label %ehcleanup58

ehcleanup58:                                      ; preds = %lpad49, %lpad44
  %ehselector.slot.1 = phi i32 [ %30, %lpad49 ], [ %27, %lpad44 ]
  %exn.slot.1 = phi i8* [ %31, %lpad49 ], [ %26, %lpad44 ]
  call void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED1Ev(%"class.std::__cxx11::basic_ostringstream"* nonnull %ss) #20
  br label %ehcleanup59

ehcleanup59:                                      ; preds = %ehcleanup58, %lpad42
  %ehselector.slot.2 = phi i32 [ %ehselector.slot.1, %ehcleanup58 ], [ %24, %lpad42 ]
  %exn.slot.2 = phi i8* [ %exn.slot.1, %ehcleanup58 ], [ %23, %lpad42 ]
  %32 = bitcast %"class.std::__cxx11::basic_ostringstream"* %ss to i8*
  call void @llvm.lifetime.end.p0i8(i64 376, i8* nonnull %32) #20
  br label %ehcleanup60

ehcleanup60:                                      ; preds = %ehcleanup59, %lpad26
  %ehselector.slot.3 = phi i32 [ %ehselector.slot.2, %ehcleanup59 ], [ %19, %lpad26 ]
  %exn.slot.3 = phi i8* [ %exn.slot.2, %ehcleanup59 ], [ %18, %lpad26 ]
  %33 = bitcast float* %psnr to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %33) #20
  br label %ehcleanup122

invoke.cont84:                                    ; preds = %for.cond.cleanup
  %conv87 = fsub float 1.000000e+02, %conv73
  invoke void @_Z17dumpFinalAccuracyf(float %conv87)
          to label %invoke.cont89 unwind label %lpad88

invoke.cont89:                                    ; preds = %invoke.cont84
  %call91 = call i32 @fclose(%struct._IO_FILE* %call10)
  %cmp94229 = icmp eq i64 %3, 0
  br i1 %cmp94229, label %for.cond.cleanup95, label %for.body96

for.cond.cleanup95:                               ; preds = %for.body96, %invoke.cont89
  %var.0.lcssa = phi float [ 0.000000e+00, %invoke.cont89 ], [ %add102, %for.body96 ]
  %div107 = fdiv float %var.0.lcssa, %conv78
  %call110 = call float @_ZSt4sqrtf(float %div107)
  invoke void @_Z11dumpPSNRStdf(float %call110)
          to label %invoke.cont111 unwind label %lpad108

lpad81:                                           ; preds = %for.cond.cleanup
  %34 = landingpad { i8*, i32 }
          cleanup
  %35 = extractvalue { i8*, i32 } %34, 0
  %36 = extractvalue { i8*, i32 } %34, 1
  br label %ehcleanup122

lpad88:                                           ; preds = %invoke.cont84
  %37 = landingpad { i8*, i32 }
          cleanup
  %38 = extractvalue { i8*, i32 } %37, 0
  %39 = extractvalue { i8*, i32 } %37, 1
  br label %ehcleanup122

for.body96:                                       ; preds = %for.body96, %invoke.cont89
  %i92.0231 = phi i64 [ %inc104, %for.body96 ], [ 0, %invoke.cont89 ]
  %var.0230 = phi float [ %add102, %for.body96 ], [ 0.000000e+00, %invoke.cont89 ]
  %call97 = call dereferenceable(4) float* @_ZNSt6vectorIfSaIfEEixEm(%"class.std::vector"* nonnull %psnr_list, i64 %i92.0231) #20
  %40 = load float, float* %call97, align 4, !tbaa !25
  %sub98 = fsub float %40, %div79
  %call99 = call dereferenceable(4) float* @_ZNSt6vectorIfSaIfEEixEm(%"class.std::vector"* nonnull %psnr_list, i64 %i92.0231) #20
  %41 = load float, float* %call99, align 4, !tbaa !25
  %sub100 = fsub float %41, %div79
  %mul101 = fmul float %sub98, %sub100
  %add102 = fadd float %var.0230, %mul101
  %inc104 = add nuw i64 %i92.0231, 1
  %exitcond = icmp eq i64 %inc104, %3
  br i1 %exitcond, label %for.cond.cleanup95, label %for.body96

invoke.cont111:                                   ; preds = %for.cond.cleanup95
  call void @_ZNSt6vectorIfSaIfEED2Ev(%"class.std::vector"* nonnull %psnr_list) #20
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #20
  ret float %conv73

lpad108:                                          ; preds = %for.cond.cleanup95
  %42 = landingpad { i8*, i32 }
          cleanup
  %43 = extractvalue { i8*, i32 } %42, 0
  %44 = extractvalue { i8*, i32 } %42, 1
  br label %ehcleanup122

ehcleanup122:                                     ; preds = %lpad108, %lpad88, %lpad81, %ehcleanup60
  %ehselector.slot.7 = phi i32 [ %ehselector.slot.3, %ehcleanup60 ], [ %36, %lpad81 ], [ %44, %lpad108 ], [ %39, %lpad88 ]
  %exn.slot.7 = phi i8* [ %exn.slot.3, %ehcleanup60 ], [ %35, %lpad81 ], [ %43, %lpad108 ], [ %38, %lpad88 ]
  call void @_ZNSt6vectorIfSaIfEED2Ev(%"class.std::vector"* nonnull %psnr_list) #20
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %0) #20
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.7, 0
  %lpad.val129 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.7, 1
  resume { i8*, i32 } %lpad.val129
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float @_ZSt5log10f(float %__x) local_unnamed_addr #12 comdat {
entry:
  %call = tail call float @log10f(float %__x) #20
  ret float %call
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float @_ZSt4sqrtf(float %__x) local_unnamed_addr #12 comdat {
entry:
  %call = tail call float @sqrtf(float %__x) #20
  ret float %call
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @_Z10dumpOutputPvPKc(i8* nocapture readonly %output_ptr, i8* nocapture readonly %file_name) local_unnamed_addr #6 {
entry:
  %size_in_bytes1 = getelementptr inbounds i8, i8* %output_ptr, i64 80
  %0 = bitcast i8* %size_in_bytes1 to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !23
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.38, i64 0, i64 0), i64 %1)
  %host_data2 = getelementptr inbounds i8, i8* %output_ptr, i64 48
  %2 = bitcast i8* %host_data2 to i8**
  %3 = load i8*, i8** %2, align 8, !tbaa !22
  %call3 = tail call %struct._IO_FILE* @fopen(i8* %file_name, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.22, i64 0, i64 0))
  %4 = tail call i64 @fwrite_unlocked(i8* %3, i64 1, i64 %1, %struct._IO_FILE* %call3)
  %call5 = tail call i32 @fclose(%struct._IO_FILE* %call3)
  ret void
}

; Function Attrs: norecurse uwtable
define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #13 {
entry:
  %call = tail call noalias i8* @malloc(i64 48) #20
  %call1 = tail call i8* @create4DTensor(i32 0, i32 0, i64 1, i64 1, i64 2, i64 2)
  %input12 = bitcast i8* %call to i8**
  store i8* %call1, i8** %input12, align 1, !tbaa !42
  %input1_bytes = getelementptr inbounds i8, i8* %call, i64 8
  %0 = bitcast i8* %input1_bytes to i64*
  store i64 0, i64* %0, align 1, !tbaa !45
  %host_data = getelementptr inbounds i8, i8* %call1, i64 48
  %1 = bitcast i8* %host_data to float**
  %2 = load float*, float** %1, align 8, !tbaa !22
  store float 0.000000e+00, float* %2, align 4, !tbaa !25
  %arrayidx3 = getelementptr inbounds float, float* %2, i64 1
  store float 0.000000e+00, float* %arrayidx3, align 4, !tbaa !25
  %arrayidx4 = getelementptr inbounds float, float* %2, i64 2
  store float -1.000000e+00, float* %arrayidx4, align 4, !tbaa !25
  %arrayidx5 = getelementptr inbounds float, float* %2, i64 3
  store float 0.000000e+00, float* %arrayidx5, align 4, !tbaa !25
  %call6 = tail call i8* @create4DTensor(i32 0, i32 0, i64 1, i64 1, i64 2, i64 2)
  %input27 = getelementptr inbounds i8, i8* %call, i64 16
  %3 = bitcast i8* %input27 to i8**
  store i8* %call6, i8** %3, align 1, !tbaa !46
  %input2_bytes = getelementptr inbounds i8, i8* %call, i64 24
  %4 = bitcast i8* %input2_bytes to i64*
  store i64 0, i64* %4, align 1, !tbaa !47
  %host_data9 = getelementptr inbounds i8, i8* %call6, i64 48
  %5 = bitcast i8* %host_data9 to float**
  %6 = load float*, float** %5, align 8, !tbaa !22
  store float 0xBFE3333340000000, float* %6, align 4, !tbaa !25
  %arrayidx11 = getelementptr inbounds float, float* %6, i64 1
  store float 0x4002666660000000, float* %arrayidx11, align 4, !tbaa !25
  %arrayidx12 = getelementptr inbounds float, float* %6, i64 2
  store float 0.000000e+00, float* %arrayidx12, align 4, !tbaa !25
  %arrayidx13 = getelementptr inbounds float, float* %6, i64 3
  store float 0.000000e+00, float* %arrayidx13, align 4, !tbaa !25
  %call14 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.39, i64 0, i64 0))
  %call15 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPKv(%"class.std::basic_ostream"* nonnull %call14, i8* %call)
  %call16 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %call15, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %call17 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.40, i64 0, i64 0))
  %call19 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPKv(%"class.std::basic_ostream"* nonnull %call17, i8* %call)
  %call20 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %call19, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %call21 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.41, i64 0, i64 0))
  %call23 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPKv(%"class.std::basic_ostream"* nonnull %call21, i8* nonnull %input27)
  %call24 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %call23, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %7 = bitcast void (i8*, i64, i8*, i64)* undef to i8*
  %8 = call i8* @GradFunction(i8* %call)
  %host_data27 = getelementptr inbounds i8, i8* %8, i64 48
  %9 = bitcast i8* %host_data27 to float**
  %10 = load float*, float** %9, align 8, !tbaa !22
  %call28 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.42, i64 0, i64 0))
  %num_elems = getelementptr inbounds i8, i8* %8, i64 72
  %11 = bitcast i8* %num_elems to i64*
  %12 = load i64, i64* %11, align 8, !tbaa !21
  %call29 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEm(%"class.std::basic_ostream"* nonnull %call28, i64 %12)
  %call30 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %call29, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %call31 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.43, i64 0, i64 0))
  %size_in_bytes = getelementptr inbounds i8, i8* %8, i64 80
  %13 = bitcast i8* %size_in_bytes to i64*
  %14 = load i64, i64* %13, align 8, !tbaa !23
  %call32 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEm(%"class.std::basic_ostream"* nonnull %call31, i64 %14)
  %call33 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %call32, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %call34 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.44, i64 0, i64 0))
  %15 = load float, float* %10, align 4, !tbaa !25
  %call36 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call34, float %15)
  %call37 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call36, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.45, i64 0, i64 0))
  %arrayidx38 = getelementptr inbounds float, float* %10, i64 1
  %16 = load float, float* %arrayidx38, align 4, !tbaa !25
  %call39 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call37, float %16)
  %call40 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call39, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.45, i64 0, i64 0))
  %arrayidx41 = getelementptr inbounds float, float* %10, i64 2
  %17 = load float, float* %arrayidx41, align 4, !tbaa !25
  %call42 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call40, float %17)
  %call43 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call42, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.45, i64 0, i64 0))
  %arrayidx44 = getelementptr inbounds float, float* %10, i64 3
  %18 = load float, float* %arrayidx44, align 4, !tbaa !25
  %call45 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call43, float %18)
  %call46 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %call45, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  call void @llvm_hpvm_initApproxhpvmRt(i32 0)
  call void @llvm_hpvm_initializeRuntimeController(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @0, i32 0, i32 0))
  tail call void @startMemTracking()
  %graph_Z4rootPvmS_m_cloned = call i8* @llvm_hpvm_cpu_launch(i8* (i8*)* @LaunchDataflowGraph, i8* %call)
  call void @llvm_hpvm_cpu_wait(i8* %graph_Z4rootPvmS_m_cloned)
  %r = getelementptr inbounds i8, i8* %call, i64 32
  %tensor = bitcast i8* %r to i8**
  %19 = load i8*, i8** %tensor, align 1, !tbaa !48
  tail call void @hpvm_request_tensor(i8* %19, i32 0)
  %host_data49 = getelementptr inbounds i8, i8* %19, i64 48
  %20 = bitcast i8* %host_data49 to float**
  %21 = load float*, float** %20, align 8, !tbaa !22
  %call50 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.42, i64 0, i64 0))
  %num_elems51 = getelementptr inbounds i8, i8* %19, i64 72
  %22 = bitcast i8* %num_elems51 to i64*
  %23 = load i64, i64* %22, align 8, !tbaa !21
  %call52 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEm(%"class.std::basic_ostream"* nonnull %call50, i64 %23)
  %call53 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %call52, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %call54 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.43, i64 0, i64 0))
  %size_in_bytes55 = getelementptr inbounds i8, i8* %19, i64 80
  %24 = bitcast i8* %size_in_bytes55 to i64*
  %25 = load i64, i64* %24, align 8, !tbaa !23
  %call56 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEm(%"class.std::basic_ostream"* nonnull %call54, i64 %25)
  %call57 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %call56, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  %call58 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) @_ZSt4cout, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.44, i64 0, i64 0))
  %26 = load float, float* %21, align 4, !tbaa !25
  %call60 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call58, float %26)
  %call61 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call60, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.45, i64 0, i64 0))
  %arrayidx62 = getelementptr inbounds float, float* %21, i64 1
  %27 = load float, float* %arrayidx62, align 4, !tbaa !25
  %call63 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call61, float %27)
  %call64 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call63, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.45, i64 0, i64 0))
  %arrayidx65 = getelementptr inbounds float, float* %21, i64 2
  %28 = load float, float* %arrayidx65, align 4, !tbaa !25
  %call66 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call64, float %28)
  %call67 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call66, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.45, i64 0, i64 0))
  %arrayidx68 = getelementptr inbounds float, float* %21, i64 3
  %29 = load float, float* %arrayidx68, align 4, !tbaa !25
  %call69 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEf(%"class.std::basic_ostream"* nonnull %call67, float %29)
  %call70 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* nonnull %call69, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* nonnull @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_)
  call void @llvm_hpvm_cleanupApproxhpvmRt()
  call void @llvm_hpvm_clearRuntimeController()
  ret i32 0
}

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc(%"class.std::basic_ostream"* dereferenceable(272) %__out, i8* %__s) local_unnamed_addr #10 {
entry:
  %tobool = icmp eq i8* %__s, null
  br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %0 = bitcast %"class.std::basic_ostream"* %__out to i8**
  %vtable = load i8*, i8** %0, align 8, !tbaa !28
  %vbase.offset.ptr = getelementptr i8, i8* %vtable, i64 -24
  %1 = bitcast i8* %vbase.offset.ptr to i64*
  %vbase.offset = load i64, i64* %1, align 8
  %2 = bitcast %"class.std::basic_ostream"* %__out to i8*
  %add.ptr = getelementptr inbounds i8, i8* %2, i64 %vbase.offset
  %3 = bitcast i8* %add.ptr to %"class.std::basic_ios"*
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate(%"class.std::basic_ios"* nonnull %3, i32 1)
  br label %if.end

if.else:                                          ; preds = %entry
  %call = tail call i64 @_ZNSt11char_traitsIcE6lengthEPKc(i8* nonnull %__s)
  %call1 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* nonnull dereferenceable(272) %__out, i8* nonnull %__s, i64 %call)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret %"class.std::basic_ostream"* %__out
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPKv(%"class.std::basic_ostream"* %this, i8* %__p) local_unnamed_addr #0 align 2 {
entry:
  %call = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIPKvEERSoT_(%"class.std::basic_ostream"* %this, i8* %__p)
  ret %"class.std::basic_ostream"* %call
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEPFRSoS_E(%"class.std::basic_ostream"* %this, %"class.std::basic_ostream"* (%"class.std::basic_ostream"*)* %__pf) local_unnamed_addr #0 align 2 {
entry:
  %call = tail call dereferenceable(272) %"class.std::basic_ostream"* %__pf(%"class.std::basic_ostream"* dereferenceable(272) %this)
  ret %"class.std::basic_ostream"* %call
}

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272) %__os) #10 {
entry:
  %0 = bitcast %"class.std::basic_ostream"* %__os to i8**
  %vtable = load i8*, i8** %0, align 8, !tbaa !28
  %vbase.offset.ptr = getelementptr i8, i8* %vtable, i64 -24
  %1 = bitcast i8* %vbase.offset.ptr to i64*
  %vbase.offset = load i64, i64* %1, align 8
  %2 = bitcast %"class.std::basic_ostream"* %__os to i8*
  %add.ptr = getelementptr inbounds i8, i8* %2, i64 %vbase.offset
  %3 = bitcast i8* %add.ptr to %"class.std::basic_ios"*
  %call = tail call signext i8 @_ZNKSt9basic_iosIcSt11char_traitsIcEE5widenEc(%"class.std::basic_ios"* nonnull %3, i8 signext 10)
  %call1 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %__os, i8 signext %call)
  %call2 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZSt5flushIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* nonnull dereferenceable(272) %call1)
  ret %"class.std::basic_ostream"* %call2
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEm(%"class.std::basic_ostream"* %this, i64 %__n) local_unnamed_addr #0 align 2 {
entry:
  %call = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertImEERSoT_(%"class.std::basic_ostream"* %this, i64 %__n)
  ret %"class.std::basic_ostream"* %call
}

declare dso_local void @startMemTracking() local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt12_Vector_baseIfSaIfEEC2Ev(%"struct.std::_Vector_base"* %this) unnamed_addr #4 comdat align 2 {
entry:
  %_M_impl = getelementptr inbounds %"struct.std::_Vector_base", %"struct.std::_Vector_base"* %this, i64 0, i32 0
  tail call void @_ZNSt12_Vector_baseIfSaIfEE12_Vector_implC2Ev(%"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl"* %_M_impl) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt12_Vector_baseIfSaIfEE12_Vector_implC2Ev(%"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl"* %this) unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl"* %this to %"class.std::allocator"*
  tail call void @_ZNSaIfEC2Ev(%"class.std::allocator"* %0) #20
  %1 = getelementptr inbounds %"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl", %"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl"* %this, i64 0, i32 0
  tail call void @_ZNSt12_Vector_baseIfSaIfEE17_Vector_impl_dataC2Ev(%"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl_data"* %1) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSaIfEC2Ev(%"class.std::allocator"* %this) unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator"* %this to %"class.__gnu_cxx::new_allocator"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIfEC2Ev(%"class.__gnu_cxx::new_allocator"* %0) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt12_Vector_baseIfSaIfEE17_Vector_impl_dataC2Ev(%"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl_data"* %this) unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"struct.std::_Vector_base<float, std::allocator<float> >::_Vector_impl_data"* %this to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 24, i1 false)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIfEC2Ev(%"class.__gnu_cxx::new_allocator"* %this) unnamed_addr #4 comdat align 2 {
entry:
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local i32 @_ZNSt8ios_base4setfESt13_Ios_FmtflagsS0_(%"class.std::ios_base"* %this, i32 %__fmtfl, i32 %__mask) local_unnamed_addr #0 comdat align 2 {
entry:
  %_M_flags = getelementptr inbounds %"class.std::ios_base", %"class.std::ios_base"* %this, i64 0, i32 3
  %0 = load i32, i32* %_M_flags, align 8, !tbaa !49
  %call = tail call i32 @_ZStcoSt13_Ios_Fmtflags(i32 %__mask)
  %call3 = tail call dereferenceable(4) i32* @_ZStaNRSt13_Ios_FmtflagsS_(i32* nonnull dereferenceable(4) %_M_flags, i32 %call)
  %call4 = tail call i32 @_ZStanSt13_Ios_FmtflagsS_(i32 %__fmtfl, i32 %__mask)
  %call6 = tail call dereferenceable(4) i32* @_ZStoRRSt13_Ios_FmtflagsS_(i32* nonnull dereferenceable(4) %_M_flags, i32 %call4)
  ret i32 %0
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local dereferenceable(4) i32* @_ZStaNRSt13_Ios_FmtflagsS_(i32* dereferenceable(4) %__a, i32 %__b) local_unnamed_addr #10 comdat {
entry:
  %0 = load i32, i32* %__a, align 4, !tbaa !55
  %call = tail call i32 @_ZStanSt13_Ios_FmtflagsS_(i32 %0, i32 %__b)
  store i32 %call, i32* %__a, align 4, !tbaa !55
  ret i32* %__a
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i32 @_ZStcoSt13_Ios_Fmtflags(i32 %__a) local_unnamed_addr #12 comdat {
entry:
  %neg = xor i32 %__a, -1
  ret i32 %neg
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local dereferenceable(4) i32* @_ZStoRRSt13_Ios_FmtflagsS_(i32* dereferenceable(4) %__a, i32 %__b) local_unnamed_addr #10 comdat {
entry:
  %0 = load i32, i32* %__a, align 4, !tbaa !55
  %call = tail call i32 @_ZStorSt13_Ios_FmtflagsS_(i32 %0, i32 %__b)
  store i32 %call, i32* %__a, align 4, !tbaa !55
  ret i32* %__a
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i32 @_ZStanSt13_Ios_FmtflagsS_(i32 %__a, i32 %__b) local_unnamed_addr #12 comdat {
entry:
  %and = and i32 %__b, %__a
  ret i32 %and
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i32 @_ZStorSt13_Ios_FmtflagsS_(i32 %__a, i32 %__b) local_unnamed_addr #12 comdat {
entry:
  %or = or i32 %__b, %__a
  ret i32 %or
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EEC2Ev(%"struct.std::_Vector_base.4"* %this) unnamed_addr #4 comdat align 2 {
entry:
  %_M_impl = getelementptr inbounds %"struct.std::_Vector_base.4", %"struct.std::_Vector_base.4"* %this, i64 0, i32 0
  tail call void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE12_Vector_implC2Ev(%"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl"* %_M_impl) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE12_Vector_implC2Ev(%"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl"* %this) unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl"* %this to %"class.std::allocator.5"*
  tail call void @_ZNSaI9ClassProbEC2Ev(%"class.std::allocator.5"* %0) #20
  %1 = getelementptr inbounds %"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl", %"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl"* %this, i64 0, i32 0
  tail call void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE17_Vector_impl_dataC2Ev(%"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl_data"* %1) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSaI9ClassProbEC2Ev(%"class.std::allocator.5"* %this) unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator.5"* %this to %"class.__gnu_cxx::new_allocator.6"*
  tail call void @_ZN9__gnu_cxx13new_allocatorI9ClassProbEC2Ev(%"class.__gnu_cxx::new_allocator.6"* %0) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE17_Vector_impl_dataC2Ev(%"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl_data"* %this) unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"struct.std::_Vector_base<ClassProb, std::allocator<ClassProb> >::_Vector_impl_data"* %this to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %0, i8 0, i64 24, i1 false)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorI9ClassProbEC2Ev(%"class.__gnu_cxx::new_allocator.6"* %this) unnamed_addr #4 comdat align 2 {
entry:
  ret void
}

; Function Attrs: nofree nounwind
declare dso_local float @log10f(float) local_unnamed_addr #7

; Function Attrs: nofree nounwind
declare dso_local float @sqrtf(float) local_unnamed_addr #7

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 {
entry:
  %_M_p = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 0, i32 0
  %0 = load i8*, i8** %_M_p, align 8, !tbaa !56
  ret i8* %0
}

; Function Attrs: noinline noreturn nounwind
define linkonce_odr hidden void @__clang_call_terminate(i8*) local_unnamed_addr #14 comdat {
  %2 = tail call i8* @__cxa_begin_catch(i8* %0) #20
  tail call void @_ZSt9terminatev() #21
  unreachable
}

declare dso_local i8* @__cxa_begin_catch(i8*) local_unnamed_addr

declare dso_local void @_ZSt9terminatev() local_unnamed_addr

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIcEC2Ev(%"class.__gnu_cxx::new_allocator.1"* %this) unnamed_addr #4 comdat align 2 {
entry:
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIcED2Ev(%"class.__gnu_cxx::new_allocator.1"* %this) unnamed_addr #4 comdat align 2 {
entry:
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_disposeEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #0 align 2 {
entry:
  %call = tail call zeroext i1 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv(%"class.std::__cxx11::basic_string"* %this)
  br i1 %call, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %_M_allocated_capacity = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 2, i32 0
  %0 = load i64, i64* %_M_allocated_capacity, align 8, !tbaa !36
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm(%"class.std::__cxx11::basic_string"* %this, i64 %0) #20
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSaIcED2Ev(%"class.std::allocator.0"* %this) unnamed_addr #4 align 2 {
entry:
  %0 = bitcast %"class.std::allocator.0"* %this to %"class.__gnu_cxx::new_allocator.1"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIcED2Ev(%"class.__gnu_cxx::new_allocator.1"* %0) #20
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local zeroext i1 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_is_localEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #0 align 2 {
entry:
  %call = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %this)
  %call2 = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv(%"class.std::__cxx11::basic_string"* %this)
  %cmp = icmp eq i8* %call, %call2
  ret i1 %cmp
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_destroyEm(%"class.std::__cxx11::basic_string"* %this, i64 %__size) local_unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = tail call dereferenceable(1) %"class.std::allocator.0"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv(%"class.std::__cxx11::basic_string"* %this)
  %call3 = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %this)
  %add = add i64 %__size, 1
  invoke void @_ZNSt16allocator_traitsISaIcEE10deallocateERS0_Pcm(%"class.std::allocator.0"* nonnull dereferenceable(1) %call, i8* %call3, i64 %add)
          to label %invoke.cont4 unwind label %lpad

invoke.cont4:                                     ; preds = %entry
  ret void

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  %1 = extractvalue { i8*, i32 } %0, 0
  tail call void @__cxa_call_unexpected(i8* %1) #21
  unreachable
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 2
  %arraydecay = bitcast %union.anon* %0 to i8*
  %call = tail call i8* @_ZNSt14pointer_traitsIPKcE10pointer_toERS0_(i8* nonnull dereferenceable(1) %arraydecay) #20
  ret i8* %call
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i8* @_ZNSt14pointer_traitsIPKcE10pointer_toERS0_(i8* dereferenceable(1) %__r) local_unnamed_addr #4 comdat align 2 {
entry:
  %call = tail call i8* @_ZSt9addressofIKcEPT_RS1_(i8* nonnull dereferenceable(1) %__r) #20
  ret i8* %call
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i8* @_ZSt9addressofIKcEPT_RS1_(i8* dereferenceable(1) %__r) local_unnamed_addr #12 comdat {
entry:
  %call = tail call i8* @_ZSt11__addressofIKcEPT_RS1_(i8* nonnull dereferenceable(1) %__r) #20
  ret i8* %call
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i8* @_ZSt11__addressofIKcEPT_RS1_(i8* dereferenceable(1) %__r) local_unnamed_addr #12 comdat {
entry:
  ret i8* %__r
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt16allocator_traitsISaIcEE10deallocateERS0_Pcm(%"class.std::allocator.0"* dereferenceable(1) %__a, i8* %__p, i64 %__n) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator.0"* %__a to %"class.__gnu_cxx::new_allocator.1"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm(%"class.__gnu_cxx::new_allocator.1"* nonnull %0, i8* %__p, i64 %__n)
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local dereferenceable(1) %"class.std::allocator.0"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 {
entry:
  %0 = bitcast %"class.std::__cxx11::basic_string"* %this to %"class.std::allocator.0"*
  ret %"class.std::allocator.0"* %0
}

declare dso_local void @__cxa_call_unexpected(i8*) local_unnamed_addr

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIcE10deallocateEPcm(%"class.__gnu_cxx::new_allocator.1"* %this, i8* %__p, i64) local_unnamed_addr #4 comdat align 2 {
entry:
  tail call void @_ZdlPv(i8* %__p) #20
  ret void
}

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdlPv(i8*) local_unnamed_addr #15

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 2
  %arraydecay = bitcast %union.anon* %0 to i8*
  %call = tail call i8* @_ZNSt14pointer_traitsIPcE10pointer_toERc(i8* nonnull dereferenceable(1) %arraydecay) #20
  ret i8* %call
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC2EPcRKS3_(%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %this, i8* %__dat, %"class.std::allocator.0"* dereferenceable(1) %__a) unnamed_addr #4 align 2 {
entry:
  %0 = bitcast %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %this to %"class.std::allocator.0"*
  tail call void @_ZNSaIcEC2ERKS_(%"class.std::allocator.0"* %0, %"class.std::allocator.0"* nonnull dereferenceable(1) %__a) #20
  %_M_p = getelementptr inbounds %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %this, i64 0, i32 0
  store i8* %__dat, i8** %_M_p, align 8, !tbaa !57
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_(%"class.std::__cxx11::basic_string"* %this, i8* %__beg, i8* %__end) local_unnamed_addr #0 comdat align 2 {
entry:
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_construct_auxIPKcEEvT_S8_St12__false_type(%"class.std::__cxx11::basic_string"* %this, i8* %__beg, i8* %__end)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt11char_traitsIcE6lengthEPKc(i8* %__s) local_unnamed_addr #4 comdat align 2 {
entry:
  %call = tail call i64 @strlen(i8* %__s) #20
  ret i64 %call
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i8* @_ZNSt14pointer_traitsIPcE10pointer_toERc(i8* dereferenceable(1) %__r) local_unnamed_addr #4 comdat align 2 {
entry:
  %call = tail call i8* @_ZSt9addressofIcEPT_RS0_(i8* nonnull dereferenceable(1) %__r) #20
  ret i8* %call
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i8* @_ZSt9addressofIcEPT_RS0_(i8* dereferenceable(1) %__r) local_unnamed_addr #12 comdat {
entry:
  %call = tail call i8* @_ZSt11__addressofIcEPT_RS0_(i8* nonnull dereferenceable(1) %__r) #20
  ret i8* %call
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i8* @_ZSt11__addressofIcEPT_RS0_(i8* dereferenceable(1) %__r) local_unnamed_addr #12 comdat {
entry:
  ret i8* %__r
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIcEC2ERKS1_(%"class.__gnu_cxx::new_allocator.1"* %this, %"class.__gnu_cxx::new_allocator.1"* dereferenceable(1)) unnamed_addr #4 comdat align 2 {
entry:
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_construct_auxIPKcEEvT_S8_St12__false_type(%"class.std::__cxx11::basic_string"* %this, i8* %__beg, i8* %__end) local_unnamed_addr #0 comdat align 2 {
entry:
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag(%"class.std::__cxx11::basic_string"* %this, i8* %__beg, i8* %__end)
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_M_constructIPKcEEvT_S8_St20forward_iterator_tag(%"class.std::__cxx11::basic_string"* %this, i8* %__beg, i8* %__end) local_unnamed_addr #0 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %__dnew = alloca i64, align 8
  %call = tail call zeroext i1 @_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_(i8* %__beg)
  %call.not = xor i1 %call, true
  %cmp = icmp eq i8* %__beg, %__end
  %or.cond = or i1 %cmp, %call.not
  br i1 %or.cond, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @_ZSt19__throw_logic_errorPKc(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.46, i64 0, i64 0)) #22
  unreachable

if.end:                                           ; preds = %entry
  %0 = bitcast i64* %__dnew to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call2 = tail call i64 @_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_(i8* %__beg, i8* %__end)
  store i64 %call2, i64* %__dnew, align 8, !tbaa !20
  %cmp3 = icmp ugt i64 %call2, 15
  br i1 %cmp3, label %if.then4, label %if.end6

if.then4:                                         ; preds = %if.end
  %call5 = call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* %this, i64* nonnull dereferenceable(8) %__dnew, i64 0)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc(%"class.std::__cxx11::basic_string"* %this, i8* %call5)
  %1 = load i64, i64* %__dnew, align 8, !tbaa !20
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm(%"class.std::__cxx11::basic_string"* %this, i64 %1)
  br label %if.end6

if.end6:                                          ; preds = %if.then4, %if.end
  %call7 = call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %this)
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_S_copy_charsEPcPKcS7_(i8* %call7, i8* %__beg, i8* %__end) #20
  %2 = load i64, i64* %__dnew, align 8, !tbaa !20
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm(%"class.std::__cxx11::basic_string"* %this, i64 %2)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local zeroext i1 @_ZN9__gnu_cxx17__is_null_pointerIKcEEbPT_(i8* %__ptr) local_unnamed_addr #12 comdat {
entry:
  %cmp = icmp eq i8* %__ptr, null
  ret i1 %cmp
}

; Function Attrs: noreturn
declare dso_local void @_ZSt19__throw_logic_errorPKc(i8*) local_unnamed_addr #16

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local i64 @_ZSt8distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_(i8* %__first, i8* %__last) local_unnamed_addr #10 comdat {
entry:
  %__first.addr = alloca i8*, align 8
  store i8* %__first, i8** %__first.addr, align 8, !tbaa !58
  call void @_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_(i8** nonnull dereferenceable(8) %__first.addr)
  %call = call i64 @_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag(i8* %__first, i8* %__last)
  ret i64 %call
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEPc(%"class.std::__cxx11::basic_string"* %this, i8* %__p) local_unnamed_addr #4 align 2 {
entry:
  %_M_p = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 0, i32 0
  store i8* %__p, i8** %_M_p, align 8, !tbaa !56
  ret void
}

declare dso_local i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"*, i64* dereferenceable(8), i64) local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE11_M_capacityEm(%"class.std::__cxx11::basic_string"* %this, i64 %__capacity) local_unnamed_addr #4 align 2 {
entry:
  %_M_allocated_capacity = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 2, i32 0
  store i64 %__capacity, i64* %_M_allocated_capacity, align 8, !tbaa !36
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_S_copy_charsEPcPKcS7_(i8* %__p, i8* %__k1, i8* %__k2) local_unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %sub.ptr.lhs.cast = ptrtoint i8* %__k2 to i64
  %sub.ptr.rhs.cast = ptrtoint i8* %__k1 to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm(i8* %__p, i8* %__k1, i64 %sub.ptr.sub)
          to label %invoke.cont unwind label %terminate.lpad

invoke.cont:                                      ; preds = %entry
  ret void

terminate.lpad:                                   ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  tail call void @__clang_call_terminate(i8* %1) #21
  unreachable
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm(%"class.std::__cxx11::basic_string"* %this, i64 %__n) local_unnamed_addr #0 align 2 {
entry:
  %ref.tmp = alloca i8, align 1
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm(%"class.std::__cxx11::basic_string"* %this, i64 %__n)
  %call = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %this)
  %arrayidx = getelementptr inbounds i8, i8* %call, i64 %__n
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %ref.tmp) #20
  store i8 0, i8* %ref.tmp, align 1, !tbaa !36
  call void @_ZNSt11char_traitsIcE6assignERcRKc(i8* dereferenceable(1) %arrayidx, i8* nonnull dereferenceable(1) %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %ref.tmp) #20
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i64 @_ZSt10__distanceIPKcENSt15iterator_traitsIT_E15difference_typeES3_S3_St26random_access_iterator_tag(i8* %__first, i8* %__last) local_unnamed_addr #12 comdat {
entry:
  %sub.ptr.lhs.cast = ptrtoint i8* %__last to i64
  %sub.ptr.rhs.cast = ptrtoint i8* %__first to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  ret i64 %sub.ptr.sub
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZSt19__iterator_categoryIPKcENSt15iterator_traitsIT_E17iterator_categoryERKS3_(i8** dereferenceable(8)) local_unnamed_addr #12 comdat {
entry:
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_S_copyEPcPKcm(i8* %__d, i8* %__s, i64 %__n) local_unnamed_addr #0 align 2 {
entry:
  %cmp = icmp eq i64 %__n, 1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call void @_ZNSt11char_traitsIcE6assignERcRKc(i8* dereferenceable(1) %__d, i8* dereferenceable(1) %__s) #20
  br label %if.end

if.else:                                          ; preds = %entry
  %call = tail call i8* @_ZNSt11char_traitsIcE4copyEPcPKcm(i8* %__d, i8* %__s, i64 %__n)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt11char_traitsIcE6assignERcRKc(i8* dereferenceable(1) %__c1, i8* dereferenceable(1) %__c2) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = load i8, i8* %__c2, align 1, !tbaa !36
  store i8 %0, i8* %__c1, align 1, !tbaa !36
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i8* @_ZNSt11char_traitsIcE4copyEPcPKcm(i8* %__s1, i8* %__s2, i64 %__n) local_unnamed_addr #4 comdat align 2 {
entry:
  %cmp = icmp eq i64 %__n, 0
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %__s1, i8* align 1 %__s2, i64 %__n, i1 false)
  br label %return

return:                                           ; preds = %if.end, %entry
  ret i8* %__s1
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_lengthEm(%"class.std::__cxx11::basic_string"* %this, i64 %__length) local_unnamed_addr #4 align 2 {
entry:
  %_M_string_length = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 1
  store i64 %__length, i64* %_M_string_length, align 8, !tbaa !30
  ret void
}

; Function Attrs: argmemonly nofree nounwind readonly
declare dso_local i64 @strlen(i8* nocapture) local_unnamed_addr #17

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZSt8_DestroyIPffEvT_S1_RSaIT0_E(float* %__first, float* %__last, %"class.std::allocator"* dereferenceable(1)) local_unnamed_addr #10 comdat {
entry:
  tail call void @_ZSt8_DestroyIPfEvT_S1_(float* %__first, float* %__last)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(1) %"class.std::allocator"* @_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"struct.std::_Vector_base"* %this to %"class.std::allocator"*
  ret %"class.std::allocator"* %0
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt12_Vector_baseIfSaIfEED2Ev(%"struct.std::_Vector_base"* %this) unnamed_addr #4 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %_M_start = getelementptr inbounds %"struct.std::_Vector_base", %"struct.std::_Vector_base"* %this, i64 0, i32 0, i32 0, i32 0
  %0 = load float*, float** %_M_start, align 8, !tbaa !6
  %_M_end_of_storage = getelementptr inbounds %"struct.std::_Vector_base", %"struct.std::_Vector_base"* %this, i64 0, i32 0, i32 0, i32 2
  %1 = bitcast float** %_M_end_of_storage to i64*
  %2 = load i64, i64* %1, align 8, !tbaa !41
  %sub.ptr.rhs.cast = ptrtoint float* %0 to i64
  %sub.ptr.sub = sub i64 %2, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 2
  invoke void @_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm(%"struct.std::_Vector_base"* %this, float* %0, i64 %sub.ptr.div)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %3 = bitcast %"struct.std::_Vector_base"* %this to %"class.__gnu_cxx::new_allocator"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIfED2Ev(%"class.__gnu_cxx::new_allocator"* %3) #20
  ret void

lpad:                                             ; preds = %entry
  %4 = landingpad { i8*, i32 }
          catch i8* null
  %5 = extractvalue { i8*, i32 } %4, 0
  %6 = bitcast %"struct.std::_Vector_base"* %this to %"class.__gnu_cxx::new_allocator"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIfED2Ev(%"class.__gnu_cxx::new_allocator"* %6) #20
  tail call void @__clang_call_terminate(i8* %5) #21
  unreachable
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZSt8_DestroyIPfEvT_S1_(float* %__first, float* %__last) local_unnamed_addr #10 comdat {
entry:
  tail call void @_ZNSt12_Destroy_auxILb1EE9__destroyIPfEEvT_S3_(float* %__first, float* %__last)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt12_Destroy_auxILb1EE9__destroyIPfEEvT_S3_(float*, float*) local_unnamed_addr #4 comdat align 2 {
entry:
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm(%"struct.std::_Vector_base"* %this, float* %__p, i64 %__n) local_unnamed_addr #0 comdat align 2 {
entry:
  %tobool = icmp eq float* %__p, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = bitcast %"struct.std::_Vector_base"* %this to %"class.std::allocator"*
  tail call void @_ZNSt16allocator_traitsISaIfEE10deallocateERS0_Pfm(%"class.std::allocator"* dereferenceable(1) %0, float* nonnull %__p, i64 %__n)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIfED2Ev(%"class.__gnu_cxx::new_allocator"* %this) unnamed_addr #4 comdat align 2 {
entry:
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt16allocator_traitsISaIfEE10deallocateERS0_Pfm(%"class.std::allocator"* dereferenceable(1) %__a, float* %__p, i64 %__n) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator"* %__a to %"class.__gnu_cxx::new_allocator"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIfE10deallocateEPfm(%"class.__gnu_cxx::new_allocator"* nonnull %0, float* %__p, i64 %__n)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIfE10deallocateEPfm(%"class.__gnu_cxx::new_allocator"* %this, float* %__p, i64) local_unnamed_addr #4 comdat align 2 {
entry:
  %1 = bitcast float* %__p to i8*
  tail call void @_ZdlPv(i8* %1) #20
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt9basic_iosIcSt11char_traitsIcEEC2Ev(%"class.std::basic_ios"* %this) unnamed_addr #4 align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %this, i64 0, i32 0
  tail call void @_ZNSt8ios_baseC2Ev(%"class.std::ios_base"* %0) #20
  %1 = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %this, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVSt9basic_iosIcSt11char_traitsIcEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %1, align 8, !tbaa !28
  %_M_tie = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %this, i64 0, i32 1
  store %"class.std::basic_ostream"* null, %"class.std::basic_ostream"** %_M_tie, align 8, !tbaa !59
  %_M_fill = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %this, i64 0, i32 2
  store i8 0, i8* %_M_fill, align 8, !tbaa !62
  %_M_fill_init = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %this, i64 0, i32 3
  store i8 0, i8* %_M_fill_init, align 1, !tbaa !63
  %_M_streambuf = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %this, i64 0, i32 4
  %2 = bitcast %"class.std::basic_streambuf"** %_M_streambuf to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %2, i8 0, i64 32, i1 false)
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSoC2Ev(%"class.std::basic_ostream"* %this, i8** %vtt) unnamed_addr #0 align 2 {
entry:
  %0 = bitcast i8** %vtt to i64*
  %1 = load i64, i64* %0, align 8
  %2 = bitcast %"class.std::basic_ostream"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !28
  %3 = getelementptr inbounds i8*, i8** %vtt, i64 1
  %4 = bitcast i8** %3 to i64*
  %5 = load i64, i64* %4, align 8
  %6 = bitcast %"class.std::basic_ostream"* %this to i8**
  %vtable.cast = inttoptr i64 %1 to i8*
  %vbase.offset.ptr = getelementptr i8, i8* %vtable.cast, i64 -24
  %7 = bitcast i8* %vbase.offset.ptr to i64*
  %vbase.offset = load i64, i64* %7, align 8
  %8 = bitcast %"class.std::basic_ostream"* %this to i8*
  %add.ptr = getelementptr inbounds i8, i8* %8, i64 %vbase.offset
  %9 = bitcast i8* %add.ptr to i64*
  store i64 %5, i64* %9, align 8, !tbaa !28
  %vtable3 = load i8*, i8** %6, align 8, !tbaa !28
  %vbase.offset.ptr4 = getelementptr i8, i8* %vtable3, i64 -24
  %10 = bitcast i8* %vbase.offset.ptr4 to i64*
  %vbase.offset5 = load i64, i64* %10, align 8
  %add.ptr6 = getelementptr inbounds i8, i8* %8, i64 %vbase.offset5
  %11 = bitcast i8* %add.ptr6 to %"class.std::basic_ios"*
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E(%"class.std::basic_ios"* %11, %"class.std::basic_streambuf"* null)
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEC2ESt13_Ios_Openmode(%"class.std::__cxx11::basic_stringbuf"* %this, i32 %__mode) unnamed_addr #0 align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::__cxx11::basic_stringbuf", %"class.std::__cxx11::basic_stringbuf"* %this, i64 0, i32 0
  tail call void @_ZNSt15basic_streambufIcSt11char_traitsIcEEC2Ev(%"class.std::basic_streambuf"* %0)
  %1 = getelementptr inbounds %"class.std::__cxx11::basic_stringbuf", %"class.std::__cxx11::basic_stringbuf"* %this, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [16 x i8*] }, { [16 x i8*] }* @_ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %1, align 8, !tbaa !28
  %_M_mode = getelementptr inbounds %"class.std::__cxx11::basic_stringbuf", %"class.std::__cxx11::basic_stringbuf"* %this, i64 0, i32 1
  store i32 %__mode, i32* %_M_mode, align 8, !tbaa !64
  %_M_string = getelementptr inbounds %"class.std::__cxx11::basic_stringbuf", %"class.std::__cxx11::basic_stringbuf"* %this, i64 0, i32 2
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2Ev(%"class.std::__cxx11::basic_string"* nonnull %_M_string) #20
  ret void
}

declare dso_local void @_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_streambufIcS1_E(%"class.std::basic_ios"*, %"class.std::basic_streambuf"*) local_unnamed_addr #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_stringbuf"* %this) unnamed_addr #12 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::__cxx11::basic_stringbuf", %"class.std::__cxx11::basic_stringbuf"* %this, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [16 x i8*] }, { [16 x i8*] }* @_ZTVNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !28
  %_M_string = getelementptr inbounds %"class.std::__cxx11::basic_stringbuf", %"class.std::__cxx11::basic_stringbuf"* %this, i64 0, i32 2
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* nonnull %_M_string) #20
  %1 = getelementptr inbounds %"class.std::__cxx11::basic_stringbuf", %"class.std::__cxx11::basic_stringbuf"* %this, i64 0, i32 0
  tail call void @_ZNSt15basic_streambufIcSt11char_traitsIcEED2Ev(%"class.std::basic_streambuf"* %1) #20
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSoD2Ev(%"class.std::basic_ostream"* %this, i8** %vtt) unnamed_addr #4 align 2 {
entry:
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt9basic_iosIcSt11char_traitsIcEED2Ev(%"class.std::basic_ios"* %this) unnamed_addr #4 align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %this, i64 0, i32 0
  tail call void @_ZNSt8ios_baseD2Ev(%"class.std::ios_base"* %0) #20
  ret void
}

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_baseC2Ev(%"class.std::ios_base"*) unnamed_addr #2

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt15basic_streambufIcSt11char_traitsIcEEC2Ev(%"class.std::basic_streambuf"* %this) unnamed_addr #4 align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::basic_streambuf", %"class.std::basic_streambuf"* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [16 x i8*] }, { [16 x i8*] }* @_ZTVSt15basic_streambufIcSt11char_traitsIcEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !28
  %_M_in_beg = getelementptr inbounds %"class.std::basic_streambuf", %"class.std::basic_streambuf"* %this, i64 0, i32 1
  %_M_buf_locale = getelementptr inbounds %"class.std::basic_streambuf", %"class.std::basic_streambuf"* %this, i64 0, i32 7
  %1 = bitcast i8** %_M_in_beg to i8*
  call void @llvm.memset.p0i8.i64(i8* nonnull align 8 %1, i8 0, i64 48, i1 false)
  tail call void @_ZNSt6localeC1Ev(%"class.std::locale"* nonnull %_M_buf_locale) #20
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2Ev(%"class.std::__cxx11::basic_string"* %this) unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %ref.tmp = alloca %"class.std::allocator.0", align 1
  %_M_dataplus = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 0
  %call = tail call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv(%"class.std::__cxx11::basic_string"* %this)
  %0 = getelementptr inbounds %"class.std::allocator.0", %"class.std::allocator.0"* %ref.tmp, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %0) #20
  call void @_ZNSaIcEC2Ev(%"class.std::allocator.0"* nonnull %ref.tmp) #20
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC2EPcOS3_(%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %_M_dataplus, i8* %call, %"class.std::allocator.0"* nonnull dereferenceable(1) %ref.tmp)
  call void @_ZNSaIcED2Ev(%"class.std::allocator.0"* nonnull %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %0) #20
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm(%"class.std::__cxx11::basic_string"* %this, i64 0)
          to label %invoke.cont4 unwind label %lpad3

invoke.cont4:                                     ; preds = %entry
  ret void

lpad3:                                            ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* null
  %2 = extractvalue { i8*, i32 } %1, 0
  %3 = bitcast %"class.std::__cxx11::basic_string"* %this to %"class.std::allocator.0"*
  call void @_ZNSaIcED2Ev(%"class.std::allocator.0"* %3) #20
  call void @__clang_call_terminate(i8* %2) #21
  unreachable
}

; Function Attrs: nounwind
declare dso_local void @_ZNSt6localeC1Ev(%"class.std::locale"*) unnamed_addr #2

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC2EPcOS3_(%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %this, i8* %__dat, %"class.std::allocator.0"* dereferenceable(1) %__a) unnamed_addr #4 align 2 {
entry:
  %0 = bitcast %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %this to %"class.std::allocator.0"*
  %call = tail call dereferenceable(1) %"class.std::allocator.0"* @_ZSt4moveIRSaIcEEONSt16remove_referenceIT_E4typeEOS3_(%"class.std::allocator.0"* nonnull dereferenceable(1) %__a) #20
  tail call void @_ZNSaIcEC2ERKS_(%"class.std::allocator.0"* %0, %"class.std::allocator.0"* nonnull dereferenceable(1) %call) #20
  %_M_p = getelementptr inbounds %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %this, i64 0, i32 0
  store i8* %__dat, i8** %_M_p, align 8, !tbaa !57
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(1) %"class.std::allocator.0"* @_ZSt4moveIRSaIcEEONSt16remove_referenceIT_E4typeEOS3_(%"class.std::allocator.0"* dereferenceable(1) %__t) local_unnamed_addr #4 comdat {
entry:
  ret %"class.std::allocator.0"* %__t
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt15basic_streambufIcSt11char_traitsIcEED2Ev(%"class.std::basic_streambuf"* %this) unnamed_addr #4 align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::basic_streambuf", %"class.std::basic_streambuf"* %this, i64 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [16 x i8*] }, { [16 x i8*] }* @_ZTVSt15basic_streambufIcSt11char_traitsIcEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8, !tbaa !28
  %_M_buf_locale = getelementptr inbounds %"class.std::basic_streambuf", %"class.std::basic_streambuf"* %this, i64 0, i32 7
  tail call void @_ZNSt6localeD1Ev(%"class.std::locale"* nonnull %_M_buf_locale) #20
  ret void
}

; Function Attrs: nounwind
declare dso_local void @_ZNSt6localeD1Ev(%"class.std::locale"*) unnamed_addr #2

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_baseD2Ev(%"class.std::ios_base"*) unnamed_addr #2

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1119basic_ostringstreamIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_ostringstream"* %this, i8** %vtt) unnamed_addr #4 align 2 {
entry:
  %0 = bitcast i8** %vtt to i64*
  %1 = load i64, i64* %0, align 8
  %2 = bitcast %"class.std::__cxx11::basic_ostringstream"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !28
  %3 = getelementptr inbounds i8*, i8** %vtt, i64 3
  %4 = bitcast i8** %3 to i64*
  %5 = load i64, i64* %4, align 8
  %vtable.cast = inttoptr i64 %1 to i8*
  %vbase.offset.ptr = getelementptr i8, i8* %vtable.cast, i64 -24
  %6 = bitcast i8* %vbase.offset.ptr to i64*
  %vbase.offset = load i64, i64* %6, align 8
  %7 = bitcast %"class.std::__cxx11::basic_ostringstream"* %this to i8*
  %add.ptr = getelementptr inbounds i8, i8* %7, i64 %vbase.offset
  %8 = bitcast i8* %add.ptr to i64*
  store i64 %5, i64* %8, align 8, !tbaa !28
  %_M_stringbuf = getelementptr inbounds %"class.std::__cxx11::basic_ostringstream", %"class.std::__cxx11::basic_ostringstream"* %this, i64 0, i32 1
  tail call void @_ZNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_stringbuf"* nonnull %_M_stringbuf) #20
  %9 = bitcast %"class.std::__cxx11::basic_ostringstream"* %this to %"class.std::basic_ostream"*
  %10 = getelementptr inbounds i8*, i8** %vtt, i64 1
  tail call void @_ZNSoD2Ev(%"class.std::basic_ostream"* %9, i8** nonnull %10) #20
  ret void
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIdEERSoT_(%"class.std::basic_ostream"*, double) local_unnamed_addr #1

; Function Attrs: uwtable
define available_externally dso_local void @_ZNKSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEE3strEv(%"class.std::__cxx11::basic_string"* noalias sret %agg.result, %"class.std::__cxx11::basic_stringbuf"* %this) local_unnamed_addr #0 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %ref.tmp = alloca %"class.std::allocator.0", align 1
  %0 = getelementptr inbounds %"class.std::allocator.0", %"class.std::allocator.0"* %ref.tmp, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %0) #20
  %_M_string = getelementptr inbounds %"class.std::__cxx11::basic_stringbuf", %"class.std::__cxx11::basic_stringbuf"* %this, i64 0, i32 2
  call void @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13get_allocatorEv(%"class.std::allocator.0"* nonnull sret %ref.tmp, %"class.std::__cxx11::basic_string"* nonnull %_M_string) #20
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2ERKS3_(%"class.std::__cxx11::basic_string"* %agg.result, %"class.std::allocator.0"* nonnull dereferenceable(1) %ref.tmp) #20
  call void @_ZNSaIcED2Ev(%"class.std::allocator.0"* nonnull %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %0) #20
  %1 = getelementptr inbounds %"class.std::__cxx11::basic_stringbuf", %"class.std::__cxx11::basic_stringbuf"* %this, i64 0, i32 0
  %call = call i8* @_ZNKSt15basic_streambufIcSt11char_traitsIcEE4pptrEv(%"class.std::basic_streambuf"* %1)
  %tobool = icmp eq i8* %call, null
  br i1 %tobool, label %if.else19, label %if.then

if.then:                                          ; preds = %entry
  %call3 = call i8* @_ZNKSt15basic_streambufIcSt11char_traitsIcEE4pptrEv(%"class.std::basic_streambuf"* %1)
  %call5 = call i8* @_ZNKSt15basic_streambufIcSt11char_traitsIcEE5egptrEv(%"class.std::basic_streambuf"* %1)
  %cmp = icmp ugt i8* %call3, %call5
  %call8 = call i8* @_ZNKSt15basic_streambufIcSt11char_traitsIcEE5pbaseEv(%"class.std::basic_streambuf"* %1)
  br i1 %cmp, label %if.then6, label %if.else

if.then6:                                         ; preds = %if.then
  %call10 = call i8* @_ZNKSt15basic_streambufIcSt11char_traitsIcEE4pptrEv(%"class.std::basic_streambuf"* %1)
  %call12 = invoke dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignIPcvEERS4_T_S8_(%"class.std::__cxx11::basic_string"* %agg.result, i8* %call8, i8* %call10)
          to label %nrvo.skipdtor unwind label %lpad

lpad:                                             ; preds = %if.else19, %if.else, %if.then6
  %2 = landingpad { i8*, i32 }
          cleanup
  call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEED2Ev(%"class.std::__cxx11::basic_string"* %agg.result) #20
  resume { i8*, i32 } %2

if.else:                                          ; preds = %if.then
  %call16 = call i8* @_ZNKSt15basic_streambufIcSt11char_traitsIcEE5egptrEv(%"class.std::basic_streambuf"* %1)
  %call18 = invoke dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignIPcvEERS4_T_S8_(%"class.std::__cxx11::basic_string"* %agg.result, i8* %call8, i8* %call16)
          to label %nrvo.skipdtor unwind label %lpad

if.else19:                                        ; preds = %entry
  %call22 = invoke dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSERKS4_(%"class.std::__cxx11::basic_string"* %agg.result, %"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %_M_string)
          to label %nrvo.skipdtor unwind label %lpad

nrvo.skipdtor:                                    ; preds = %if.else19, %if.else, %if.then6
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13get_allocatorEv(%"class.std::allocator.0"* noalias sret %agg.result, %"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %call = tail call dereferenceable(1) %"class.std::allocator.0"* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv(%"class.std::__cxx11::basic_string"* %this)
  tail call void @_ZNSaIcEC2ERKS_(%"class.std::allocator.0"* %agg.result, %"class.std::allocator.0"* nonnull dereferenceable(1) %call) #20
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2ERKS3_(%"class.std::__cxx11::basic_string"* %this, %"class.std::allocator.0"* dereferenceable(1) %__a) unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %_M_dataplus = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 0
  %call = tail call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_local_dataEv(%"class.std::__cxx11::basic_string"* %this)
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderC2EPcRKS3_(%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider"* %_M_dataplus, i8* %call, %"class.std::allocator.0"* nonnull dereferenceable(1) %__a)
  invoke void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE13_M_set_lengthEm(%"class.std::__cxx11::basic_string"* %this, i64 0)
          to label %invoke.cont3 unwind label %lpad

invoke.cont3:                                     ; preds = %entry
  ret void

lpad:                                             ; preds = %entry
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = bitcast %"class.std::__cxx11::basic_string"* %this to %"class.std::allocator.0"*
  tail call void @_ZNSaIcED2Ev(%"class.std::allocator.0"* %2) #20
  tail call void @__clang_call_terminate(i8* %1) #21
  unreachable
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNKSt15basic_streambufIcSt11char_traitsIcEE4pptrEv(%"class.std::basic_streambuf"* %this) local_unnamed_addr #4 align 2 {
entry:
  %_M_out_cur = getelementptr inbounds %"class.std::basic_streambuf", %"class.std::basic_streambuf"* %this, i64 0, i32 5
  %0 = load i8*, i8** %_M_out_cur, align 8, !tbaa !67
  ret i8* %0
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNKSt15basic_streambufIcSt11char_traitsIcEE5egptrEv(%"class.std::basic_streambuf"* %this) local_unnamed_addr #4 align 2 {
entry:
  %_M_in_end = getelementptr inbounds %"class.std::basic_streambuf", %"class.std::basic_streambuf"* %this, i64 0, i32 3
  %0 = load i8*, i8** %_M_in_end, align 8, !tbaa !69
  ret i8* %0
}

; Function Attrs: uwtable
define linkonce_odr dso_local dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignIPcvEERS4_T_S8_(%"class.std::__cxx11::basic_string"* %this, i8* %__first, i8* %__last) local_unnamed_addr #0 comdat align 2 {
entry:
  %agg.tmp = alloca %"class.__gnu_cxx::__normal_iterator.8", align 8
  %ref.tmp = alloca %"class.__gnu_cxx::__normal_iterator.9", align 8
  %agg.tmp2 = alloca %"class.__gnu_cxx::__normal_iterator.8", align 8
  %ref.tmp3 = alloca %"class.__gnu_cxx::__normal_iterator.9", align 8
  %0 = bitcast %"class.__gnu_cxx::__normal_iterator.9"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call = tail call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5beginEv(%"class.std::__cxx11::basic_string"* %this) #20
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.9", %"class.__gnu_cxx::__normal_iterator.9"* %ref.tmp, i64 0, i32 0
  store i8* %call, i8** %coerce.dive, align 8
  call void @_ZN9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEC2IPcEERKNS0_IT_NS_11__enable_ifIXsr3std10__are_sameISC_SB_EE7__valueES8_E6__typeEEE(%"class.__gnu_cxx::__normal_iterator.8"* nonnull %agg.tmp, %"class.__gnu_cxx::__normal_iterator.9"* nonnull dereferenceable(8) %ref.tmp) #20
  %1 = bitcast %"class.__gnu_cxx::__normal_iterator.9"* %ref.tmp3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #20
  %call4 = call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE3endEv(%"class.std::__cxx11::basic_string"* %this) #20
  %coerce.dive5 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.9", %"class.__gnu_cxx::__normal_iterator.9"* %ref.tmp3, i64 0, i32 0
  store i8* %call4, i8** %coerce.dive5, align 8
  call void @_ZN9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEC2IPcEERKNS0_IT_NS_11__enable_ifIXsr3std10__are_sameISC_SB_EE7__valueES8_E6__typeEEE(%"class.__gnu_cxx::__normal_iterator.8"* nonnull %agg.tmp2, %"class.__gnu_cxx::__normal_iterator.9"* nonnull dereferenceable(8) %ref.tmp3) #20
  %coerce.dive6 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.8", %"class.__gnu_cxx::__normal_iterator.8"* %agg.tmp, i64 0, i32 0
  %2 = load i8*, i8** %coerce.dive6, align 8
  %coerce.dive7 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.8", %"class.__gnu_cxx::__normal_iterator.8"* %agg.tmp2, i64 0, i32 0
  %3 = load i8*, i8** %coerce.dive7, align 8
  %call8 = call dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7replaceEN9__gnu_cxx17__normal_iteratorIPKcS4_EES9_PcSA_(%"class.std::__cxx11::basic_string"* %this, i8* %2, i8* %3, i8* %__first, i8* %__last)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  ret %"class.std::__cxx11::basic_string"* %call8
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNKSt15basic_streambufIcSt11char_traitsIcEE5pbaseEv(%"class.std::basic_streambuf"* %this) local_unnamed_addr #4 align 2 {
entry:
  %_M_out_beg = getelementptr inbounds %"class.std::basic_streambuf", %"class.std::basic_streambuf"* %this, i64 0, i32 4
  %0 = load i8*, i8** %_M_out_beg, align 8, !tbaa !70
  ret i8* %0
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEaSERKS4_(%"class.std::__cxx11::basic_string"* %this, %"class.std::__cxx11::basic_string"* dereferenceable(32) %__str) local_unnamed_addr #0 align 2 {
entry:
  %call = tail call dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignERKS4_(%"class.std::__cxx11::basic_string"* %this, %"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %__str)
  ret %"class.std::__cxx11::basic_string"* %call
}

; Function Attrs: nounwind uwtable
define available_externally dso_local dereferenceable(1) %"class.std::allocator.0"* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE16_M_get_allocatorEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 {
entry:
  %0 = bitcast %"class.std::__cxx11::basic_string"* %this to %"class.std::allocator.0"*
  ret %"class.std::allocator.0"* %0
}

; Function Attrs: nounwind uwtable
define available_externally dso_local void @_ZNSaIcEC2ERKS_(%"class.std::allocator.0"* %this, %"class.std::allocator.0"* dereferenceable(1) %__a) unnamed_addr #4 align 2 {
entry:
  %0 = bitcast %"class.std::allocator.0"* %this to %"class.__gnu_cxx::new_allocator.1"*
  %1 = bitcast %"class.std::allocator.0"* %__a to %"class.__gnu_cxx::new_allocator.1"*
  tail call void @_ZN9__gnu_cxx13new_allocatorIcEC2ERKS1_(%"class.__gnu_cxx::new_allocator.1"* %0, %"class.__gnu_cxx::new_allocator.1"* nonnull dereferenceable(1) %1) #20
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7replaceEN9__gnu_cxx17__normal_iteratorIPKcS4_EES9_PcSA_(%"class.std::__cxx11::basic_string"* %this, i8* %__i1.coerce, i8* %__i2.coerce, i8* %__k1, i8* %__k2) local_unnamed_addr #0 align 2 {
entry:
  %__i1 = alloca %"class.__gnu_cxx::__normal_iterator.8", align 8
  %__i2 = alloca %"class.__gnu_cxx::__normal_iterator.8", align 8
  %ref.tmp = alloca %"class.__gnu_cxx::__normal_iterator.9", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.8", %"class.__gnu_cxx::__normal_iterator.8"* %__i1, i64 0, i32 0
  store i8* %__i1.coerce, i8** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.8", %"class.__gnu_cxx::__normal_iterator.8"* %__i2, i64 0, i32 0
  store i8* %__i2.coerce, i8** %coerce.dive1, align 8
  %0 = bitcast %"class.__gnu_cxx::__normal_iterator.9"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call = tail call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5beginEv(%"class.std::__cxx11::basic_string"* %this) #20
  %coerce.dive3 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.9", %"class.__gnu_cxx::__normal_iterator.9"* %ref.tmp, i64 0, i32 0
  store i8* %call, i8** %coerce.dive3, align 8
  %call4 = call i64 @_ZN9__gnu_cxxmiIPKcPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEDTmicldtfp_4baseEcldtfp0_4baseEERKNS_17__normal_iteratorIT_T1_EERKNSB_IT0_SD_EE(%"class.__gnu_cxx::__normal_iterator.8"* nonnull dereferenceable(8) %__i1, %"class.__gnu_cxx::__normal_iterator.9"* nonnull dereferenceable(8) %ref.tmp) #20
  %call5 = call i64 @_ZN9__gnu_cxxmiIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKSC_SF_(%"class.__gnu_cxx::__normal_iterator.8"* nonnull dereferenceable(8) %__i2, %"class.__gnu_cxx::__normal_iterator.8"* nonnull dereferenceable(8) %__i1) #20
  %sub.ptr.lhs.cast = ptrtoint i8* %__k2 to i64
  %sub.ptr.rhs.cast = ptrtoint i8* %__k1 to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %call6 = call dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7replaceEmmPKcm(%"class.std::__cxx11::basic_string"* %this, i64 %call4, i64 %call5, i8* %__k1, i64 %sub.ptr.sub)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  ret %"class.std::__cxx11::basic_string"* %call6
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE5beginEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %retval = alloca %"class.__gnu_cxx::__normal_iterator.9", align 8
  %ref.tmp = alloca i8*, align 8
  %0 = bitcast i8** %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %this)
  store i8* %call, i8** %ref.tmp, align 8, !tbaa !58
  call void @_ZN9__gnu_cxx17__normal_iteratorIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEC2ERKS1_(%"class.__gnu_cxx::__normal_iterator.9"* nonnull %retval, i8** nonnull dereferenceable(8) %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.9", %"class.__gnu_cxx::__normal_iterator.9"* %retval, i64 0, i32 0
  %1 = load i8*, i8** %coerce.dive, align 8
  ret i8* %1
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEC2IPcEERKNS0_IT_NS_11__enable_ifIXsr3std10__are_sameISC_SB_EE7__valueES8_E6__typeEEE(%"class.__gnu_cxx::__normal_iterator.8"* %this, %"class.__gnu_cxx::__normal_iterator.9"* dereferenceable(8) %__i) unnamed_addr #4 comdat align 2 {
entry:
  %call = tail call dereferenceable(8) i8** @_ZNK9__gnu_cxx17__normal_iteratorIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.9"* nonnull %__i) #20
  %0 = bitcast i8** %call to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %2 = bitcast %"class.__gnu_cxx::__normal_iterator.8"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !71
  ret void
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE3endEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %retval = alloca %"class.__gnu_cxx::__normal_iterator.9", align 8
  %ref.tmp = alloca i8*, align 8
  %0 = bitcast i8** %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call = tail call i8* @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7_M_dataEv(%"class.std::__cxx11::basic_string"* %this)
  %call2 = tail call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv(%"class.std::__cxx11::basic_string"* %this) #20
  %add.ptr = getelementptr inbounds i8, i8* %call, i64 %call2
  store i8* %add.ptr, i8** %ref.tmp, align 8, !tbaa !58
  call void @_ZN9__gnu_cxx17__normal_iteratorIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEC2ERKS1_(%"class.__gnu_cxx::__normal_iterator.9"* nonnull %retval, i8** nonnull dereferenceable(8) %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.9", %"class.__gnu_cxx::__normal_iterator.9"* %retval, i64 0, i32 0
  %1 = load i8*, i8** %coerce.dive, align 8
  ret i8* %1
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE7replaceEmmPKcm(%"class.std::__cxx11::basic_string"* %this, i64 %__pos, i64 %__n1, i8* %__s, i64 %__n2) local_unnamed_addr #0 align 2 {
entry:
  %call = tail call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8_M_checkEmPKc(%"class.std::__cxx11::basic_string"* %this, i64 %__pos, i8* getelementptr inbounds ([22 x i8], [22 x i8]* @.str.47, i64 0, i64 0))
  %call2 = tail call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8_M_limitEmm(%"class.std::__cxx11::basic_string"* %this, i64 %__pos, i64 %__n1) #20
  %call3 = tail call dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm(%"class.std::__cxx11::basic_string"* %this, i64 %call, i64 %call2, i8* %__s, i64 %__n2)
  ret %"class.std::__cxx11::basic_string"* %call3
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i64 @_ZN9__gnu_cxxmiIPKcPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEEDTmicldtfp_4baseEcldtfp0_4baseEERKNS_17__normal_iteratorIT_T1_EERKNSB_IT0_SD_EE(%"class.__gnu_cxx::__normal_iterator.8"* dereferenceable(8) %__lhs, %"class.__gnu_cxx::__normal_iterator.9"* dereferenceable(8) %__rhs) local_unnamed_addr #12 comdat {
entry:
  %call = tail call dereferenceable(8) i8** @_ZNK9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.8"* nonnull %__lhs) #20
  %0 = bitcast i8** %call to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %call1 = tail call dereferenceable(8) i8** @_ZNK9__gnu_cxx17__normal_iteratorIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.9"* nonnull %__rhs) #20
  %2 = bitcast i8** %call1 to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !58
  %sub.ptr.sub = sub i64 %1, %3
  ret i64 %sub.ptr.sub
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i64 @_ZN9__gnu_cxxmiIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKSC_SF_(%"class.__gnu_cxx::__normal_iterator.8"* dereferenceable(8) %__lhs, %"class.__gnu_cxx::__normal_iterator.8"* dereferenceable(8) %__rhs) local_unnamed_addr #12 comdat {
entry:
  %call = tail call dereferenceable(8) i8** @_ZNK9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.8"* nonnull %__lhs) #20
  %0 = bitcast i8** %call to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %call1 = tail call dereferenceable(8) i8** @_ZNK9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.8"* nonnull %__rhs) #20
  %2 = bitcast i8** %call1 to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !58
  %sub.ptr.sub = sub i64 %1, %3
  ret i64 %sub.ptr.sub
}

declare dso_local dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE10_M_replaceEmmPKcm(%"class.std::__cxx11::basic_string"*, i64, i64, i8*, i64) local_unnamed_addr #1

; Function Attrs: uwtable
define available_externally dso_local i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8_M_checkEmPKc(%"class.std::__cxx11::basic_string"* %this, i64 %__pos, i8* %__s) local_unnamed_addr #0 align 2 {
entry:
  %call = tail call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv(%"class.std::__cxx11::basic_string"* %this) #20
  %cmp = icmp ult i64 %call, %__pos
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call2 = tail call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv(%"class.std::__cxx11::basic_string"* %this) #20
  tail call void (i8*, ...) @_ZSt24__throw_out_of_range_fmtPKcz(i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.48, i64 0, i64 0), i8* %__s, i64 %__pos, i64 %call2) #22
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %__pos
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE8_M_limitEmm(%"class.std::__cxx11::basic_string"* %this, i64 %__pos, i64 %__off) local_unnamed_addr #4 align 2 {
entry:
  %call = tail call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv(%"class.std::__cxx11::basic_string"* %this) #20
  %sub = sub i64 %call, %__pos
  %cmp = icmp ugt i64 %sub, %__off
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %call2 = tail call i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv(%"class.std::__cxx11::basic_string"* %this) #20
  %sub3 = sub i64 %call2, %__pos
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %entry
  %cond = phi i64 [ %sub3, %cond.false ], [ %__off, %entry ]
  ret i64 %cond
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i64 @_ZNKSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4sizeEv(%"class.std::__cxx11::basic_string"* %this) local_unnamed_addr #4 align 2 {
entry:
  %_M_string_length = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %this, i64 0, i32 1
  %0 = load i64, i64* %_M_string_length, align 8, !tbaa !30
  ret i64 %0
}

; Function Attrs: noreturn
declare dso_local void @_ZSt24__throw_out_of_range_fmtPKcz(i8*, ...) local_unnamed_addr #16

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) i8** @_ZNK9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.8"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_current = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.8", %"class.__gnu_cxx::__normal_iterator.8"* %this, i64 0, i32 0
  ret i8** %_M_current
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) i8** @_ZNK9__gnu_cxx17__normal_iteratorIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.9"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_current = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.9", %"class.__gnu_cxx::__normal_iterator.9"* %this, i64 0, i32 0
  ret i8** %_M_current
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx17__normal_iteratorIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEC2ERKS1_(%"class.__gnu_cxx::__normal_iterator.9"* %this, i8** dereferenceable(8) %__i) unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast i8** %__i to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %2 = bitcast %"class.__gnu_cxx::__normal_iterator.9"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !73
  ret void
}

; Function Attrs: uwtable
define available_externally dso_local dereferenceable(32) %"class.std::__cxx11::basic_string"* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE6assignERKS4_(%"class.std::__cxx11::basic_string"* %this, %"class.std::__cxx11::basic_string"* dereferenceable(32) %__str) local_unnamed_addr #0 align 2 {
entry:
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(%"class.std::__cxx11::basic_string"* %this, %"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %__str)
  ret %"class.std::__cxx11::basic_string"* %this
}

declare dso_local void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_assignERKS4_(%"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"* dereferenceable(32)) local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZSt8_DestroyIP9ClassProbS0_EvT_S2_RSaIT0_E(%struct.ClassProb* %__first, %struct.ClassProb* %__last, %"class.std::allocator.5"* dereferenceable(1)) local_unnamed_addr #10 comdat {
entry:
  tail call void @_ZSt8_DestroyIP9ClassProbEvT_S2_(%struct.ClassProb* %__first, %struct.ClassProb* %__last)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(1) %"class.std::allocator.5"* @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base.4"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"struct.std::_Vector_base.4"* %this to %"class.std::allocator.5"*
  ret %"class.std::allocator.5"* %0
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EED2Ev(%"struct.std::_Vector_base.4"* %this) unnamed_addr #4 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %_M_start = getelementptr inbounds %"struct.std::_Vector_base.4", %"struct.std::_Vector_base.4"* %this, i64 0, i32 0, i32 0, i32 0
  %0 = load %struct.ClassProb*, %struct.ClassProb** %_M_start, align 8, !tbaa !40
  %_M_end_of_storage = getelementptr inbounds %"struct.std::_Vector_base.4", %"struct.std::_Vector_base.4"* %this, i64 0, i32 0, i32 0, i32 2
  %1 = bitcast %struct.ClassProb** %_M_end_of_storage to i64*
  %2 = load i64, i64* %1, align 8, !tbaa !39
  %sub.ptr.rhs.cast = ptrtoint %struct.ClassProb* %0 to i64
  %sub.ptr.sub = sub i64 %2, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 3
  invoke void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE13_M_deallocateEPS0_m(%"struct.std::_Vector_base.4"* %this, %struct.ClassProb* %0, i64 %sub.ptr.div)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  %3 = bitcast %"struct.std::_Vector_base.4"* %this to %"class.__gnu_cxx::new_allocator.6"*
  tail call void @_ZN9__gnu_cxx13new_allocatorI9ClassProbED2Ev(%"class.__gnu_cxx::new_allocator.6"* %3) #20
  ret void

lpad:                                             ; preds = %entry
  %4 = landingpad { i8*, i32 }
          catch i8* null
  %5 = extractvalue { i8*, i32 } %4, 0
  %6 = bitcast %"struct.std::_Vector_base.4"* %this to %"class.__gnu_cxx::new_allocator.6"*
  tail call void @_ZN9__gnu_cxx13new_allocatorI9ClassProbED2Ev(%"class.__gnu_cxx::new_allocator.6"* %6) #20
  tail call void @__clang_call_terminate(i8* %5) #21
  unreachable
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZSt8_DestroyIP9ClassProbEvT_S2_(%struct.ClassProb* %__first, %struct.ClassProb* %__last) local_unnamed_addr #10 comdat {
entry:
  tail call void @_ZNSt12_Destroy_auxILb1EE9__destroyIP9ClassProbEEvT_S4_(%struct.ClassProb* %__first, %struct.ClassProb* %__last)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt12_Destroy_auxILb1EE9__destroyIP9ClassProbEEvT_S4_(%struct.ClassProb*, %struct.ClassProb*) local_unnamed_addr #4 comdat align 2 {
entry:
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE13_M_deallocateEPS0_m(%"struct.std::_Vector_base.4"* %this, %struct.ClassProb* %__p, i64 %__n) local_unnamed_addr #0 comdat align 2 {
entry:
  %tobool = icmp eq %struct.ClassProb* %__p, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = bitcast %"struct.std::_Vector_base.4"* %this to %"class.std::allocator.5"*
  tail call void @_ZNSt16allocator_traitsISaI9ClassProbEE10deallocateERS1_PS0_m(%"class.std::allocator.5"* dereferenceable(1) %0, %struct.ClassProb* nonnull %__p, i64 %__n)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorI9ClassProbED2Ev(%"class.__gnu_cxx::new_allocator.6"* %this) unnamed_addr #4 comdat align 2 {
entry:
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt16allocator_traitsISaI9ClassProbEE10deallocateERS1_PS0_m(%"class.std::allocator.5"* dereferenceable(1) %__a, %struct.ClassProb* %__p, i64 %__n) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator.5"* %__a to %"class.__gnu_cxx::new_allocator.6"*
  tail call void @_ZN9__gnu_cxx13new_allocatorI9ClassProbE10deallocateEPS1_m(%"class.__gnu_cxx::new_allocator.6"* nonnull %0, %struct.ClassProb* %__p, i64 %__n)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorI9ClassProbE10deallocateEPS1_m(%"class.__gnu_cxx::new_allocator.6"* %this, %struct.ClassProb* %__p, i64) local_unnamed_addr #4 comdat align 2 {
entry:
  %1 = bitcast %struct.ClassProb* %__p to i8*
  tail call void @_ZdlPv(i8* %1) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt16allocator_traitsISaI9ClassProbEE9constructIS0_JRKS0_EEEvRS1_PT_DpOT0_(%"class.std::allocator.5"* dereferenceable(1) %__a, %struct.ClassProb* %__p, %struct.ClassProb* dereferenceable(8) %__args) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator.5"* %__a to %"class.__gnu_cxx::new_allocator.6"*
  %call = tail call dereferenceable(8) %struct.ClassProb* @_ZSt7forwardIRK9ClassProbEOT_RNSt16remove_referenceIS3_E4typeE(%struct.ClassProb* nonnull dereferenceable(8) %__args) #20
  tail call void @_ZN9__gnu_cxx13new_allocatorI9ClassProbE9constructIS1_JRKS1_EEEvPT_DpOT0_(%"class.__gnu_cxx::new_allocator.6"* nonnull %0, %struct.ClassProb* %__p, %struct.ClassProb* nonnull dereferenceable(8) %call) #20
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt6vectorI9ClassProbSaIS0_EE17_M_realloc_insertIJRKS0_EEEvN9__gnu_cxx17__normal_iteratorIPS0_S2_EEDpOT_(%"class.std::vector.3"* %this, %struct.ClassProb* %__position.coerce, %struct.ClassProb* dereferenceable(8) %__args) local_unnamed_addr #0 comdat align 2 {
entry:
  %__position = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %ref.tmp = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__position, i64 0, i32 0
  store %struct.ClassProb* %__position.coerce, %struct.ClassProb** %coerce.dive, align 8
  %call = tail call i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE12_M_check_lenEmPKc(%"class.std::vector.3"* %this, i64 1, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.49, i64 0, i64 0))
  %0 = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0
  %_M_start = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 0
  %1 = load %struct.ClassProb*, %struct.ClassProb** %_M_start, align 8, !tbaa !40
  %_M_finish = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 1
  %2 = load %struct.ClassProb*, %struct.ClassProb** %_M_finish, align 8, !tbaa !37
  %3 = bitcast %"class.__gnu_cxx::__normal_iterator"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3) #20
  %call3 = tail call %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE5beginEv(%"class.std::vector.3"* %this) #20
  %coerce.dive4 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %ref.tmp, i64 0, i32 0
  store %struct.ClassProb* %call3, %struct.ClassProb** %coerce.dive4, align 8
  %call5 = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__position, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3) #20
  %call6 = call %struct.ClassProb* @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE11_M_allocateEm(%"struct.std::_Vector_base.4"* %0, i64 %call)
  %4 = bitcast %"class.std::vector.3"* %this to %"class.std::allocator.5"*
  %add.ptr = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %call6, i64 %call5
  %call8 = call dereferenceable(8) %struct.ClassProb* @_ZSt7forwardIRK9ClassProbEOT_RNSt16remove_referenceIS3_E4typeE(%struct.ClassProb* nonnull dereferenceable(8) %__args) #20
  call void @_ZNSt16allocator_traitsISaI9ClassProbEE9constructIS0_JRKS0_EEEvRS1_PT_DpOT0_(%"class.std::allocator.5"* dereferenceable(1) %4, %struct.ClassProb* %add.ptr, %struct.ClassProb* nonnull dereferenceable(8) %call8) #20
  %call9 = call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__position) #20
  %5 = load %struct.ClassProb*, %struct.ClassProb** %call9, align 8, !tbaa !58
  %call10 = call dereferenceable(1) %"class.std::allocator.5"* @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base.4"* %0) #20
  %call11 = call %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE11_S_relocateEPS0_S3_S3_RS1_(%struct.ClassProb* %1, %struct.ClassProb* %5, %struct.ClassProb* %call6, %"class.std::allocator.5"* nonnull dereferenceable(1) %call10) #20
  %incdec.ptr = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %call11, i64 1
  %call12 = call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__position) #20
  %6 = load %struct.ClassProb*, %struct.ClassProb** %call12, align 8, !tbaa !58
  %call13 = call dereferenceable(1) %"class.std::allocator.5"* @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base.4"* %0) #20
  %call14 = call %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE11_S_relocateEPS0_S3_S3_RS1_(%struct.ClassProb* %6, %struct.ClassProb* %2, %struct.ClassProb* nonnull %incdec.ptr, %"class.std::allocator.5"* nonnull dereferenceable(1) %call13) #20
  %_M_end_of_storage = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 2
  %7 = bitcast %struct.ClassProb** %_M_end_of_storage to i64*
  %8 = load i64, i64* %7, align 8, !tbaa !39
  %sub.ptr.rhs.cast = ptrtoint %struct.ClassProb* %1 to i64
  %sub.ptr.sub = sub i64 %8, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 3
  call void @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE13_M_deallocateEPS0_m(%"struct.std::_Vector_base.4"* %0, %struct.ClassProb* %1, i64 %sub.ptr.div)
  store %struct.ClassProb* %call6, %struct.ClassProb** %_M_start, align 8, !tbaa !40
  store %struct.ClassProb* %call14, %struct.ClassProb** %_M_finish, align 8, !tbaa !37
  %add.ptr20 = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %call6, i64 %call
  store %struct.ClassProb* %add.ptr20, %struct.ClassProb** %_M_end_of_storage, align 8, !tbaa !39
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorI9ClassProbE9constructIS1_JRKS1_EEEvPT_DpOT0_(%"class.__gnu_cxx::new_allocator.6"* %this, %struct.ClassProb* %__p, %struct.ClassProb* dereferenceable(8) %__args) local_unnamed_addr #4 comdat align 2 {
entry:
  %call = tail call dereferenceable(8) %struct.ClassProb* @_ZSt7forwardIRK9ClassProbEOT_RNSt16remove_referenceIS3_E4typeE(%struct.ClassProb* nonnull dereferenceable(8) %__args) #20
  %0 = bitcast %struct.ClassProb* %call to i64*
  %1 = bitcast %struct.ClassProb* %__p to i64*
  %2 = load i64, i64* %0, align 4
  store i64 %2, i64* %1, align 4
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) %struct.ClassProb* @_ZSt7forwardIRK9ClassProbEOT_RNSt16remove_referenceIS3_E4typeE(%struct.ClassProb* dereferenceable(8) %__t) local_unnamed_addr #4 comdat {
entry:
  ret %struct.ClassProb* %__t
}

; Function Attrs: uwtable
define linkonce_odr dso_local i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE12_M_check_lenEmPKc(%"class.std::vector.3"* %this, i64 %__n, i8* %__s) local_unnamed_addr #0 comdat align 2 {
entry:
  %__n.addr = alloca i64, align 8
  %ref.tmp = alloca i64, align 8
  store i64 %__n, i64* %__n.addr, align 8, !tbaa !20
  %call = tail call i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE8max_sizeEv(%"class.std::vector.3"* %this) #20
  %call2 = tail call i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE4sizeEv(%"class.std::vector.3"* %this) #20
  %sub = sub i64 %call, %call2
  %cmp = icmp ult i64 %sub, %__n
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @_ZSt20__throw_length_errorPKc(i8* %__s) #22
  unreachable

if.end:                                           ; preds = %entry
  %call3 = tail call i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE4sizeEv(%"class.std::vector.3"* %this) #20
  %0 = bitcast i64* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call4 = tail call i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE4sizeEv(%"class.std::vector.3"* %this) #20
  store i64 %call4, i64* %ref.tmp, align 8, !tbaa !20
  %call5 = call dereferenceable(8) i64* @_ZSt3maxImERKT_S2_S2_(i64* nonnull dereferenceable(8) %ref.tmp, i64* nonnull dereferenceable(8) %__n.addr)
  %1 = load i64, i64* %call5, align 8, !tbaa !20
  %add = add i64 %1, %call3
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  %call6 = call i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE4sizeEv(%"class.std::vector.3"* %this) #20
  %cmp7 = icmp ult i64 %add, %call6
  br i1 %cmp7, label %cond.true, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.end
  %call8 = call i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE8max_sizeEv(%"class.std::vector.3"* %this) #20
  %cmp9 = icmp ugt i64 %add, %call8
  br i1 %cmp9, label %cond.true, label %cond.end

cond.true:                                        ; preds = %lor.lhs.false, %if.end
  %call10 = call i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE8max_sizeEv(%"class.std::vector.3"* %this) #20
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %lor.lhs.false
  %cond = phi i64 [ %call10, %cond.true ], [ %add, %lor.lhs.false ]
  ret i64 %cond
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* dereferenceable(8) %__lhs, %"class.__gnu_cxx::__normal_iterator"* dereferenceable(8) %__rhs) local_unnamed_addr #12 comdat {
entry:
  %call = tail call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__lhs) #20
  %0 = bitcast %struct.ClassProb** %call to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %call1 = tail call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__rhs) #20
  %2 = bitcast %struct.ClassProb** %call1 to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !58
  %sub.ptr.sub = sub i64 %1, %3
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 3
  ret i64 %sub.ptr.div
}

; Function Attrs: uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZNSt12_Vector_baseI9ClassProbSaIS0_EE11_M_allocateEm(%"struct.std::_Vector_base.4"* %this, i64 %__n) local_unnamed_addr #0 comdat align 2 {
entry:
  %cmp = icmp eq i64 %__n, 0
  br i1 %cmp, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = bitcast %"struct.std::_Vector_base.4"* %this to %"class.std::allocator.5"*
  %call = tail call %struct.ClassProb* @_ZNSt16allocator_traitsISaI9ClassProbEE8allocateERS1_m(%"class.std::allocator.5"* dereferenceable(1) %0, i64 %__n)
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %entry
  %cond = phi %struct.ClassProb* [ %call, %cond.true ], [ null, %entry ]
  ret %struct.ClassProb* %cond
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE11_S_relocateEPS0_S3_S3_RS1_(%struct.ClassProb* %__first, %struct.ClassProb* %__last, %struct.ClassProb* %__result, %"class.std::allocator.5"* dereferenceable(1) %__alloc) local_unnamed_addr #4 comdat align 2 {
entry:
  %call = tail call %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE14_S_do_relocateEPS0_S3_S3_RS1_St17integral_constantIbLb1EE(%struct.ClassProb* %__first, %struct.ClassProb* %__last, %struct.ClassProb* %__result, %"class.std::allocator.5"* nonnull dereferenceable(1) %__alloc) #20
  ret %struct.ClassProb* %call
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_current = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %this, i64 0, i32 0
  ret %struct.ClassProb** %_M_current
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE8max_sizeEv(%"class.std::vector.3"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0
  %call = tail call dereferenceable(1) %"class.std::allocator.5"* @_ZNKSt12_Vector_baseI9ClassProbSaIS0_EE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base.4"* %0) #20
  %call2 = tail call i64 @_ZNSt6vectorI9ClassProbSaIS0_EE11_S_max_sizeERKS1_(%"class.std::allocator.5"* nonnull dereferenceable(1) %call) #20
  ret i64 %call2
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNKSt6vectorI9ClassProbSaIS0_EE4sizeEv(%"class.std::vector.3"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_finish = getelementptr inbounds %"class.std::vector.3", %"class.std::vector.3"* %this, i64 0, i32 0, i32 0, i32 0, i32 1
  %0 = bitcast %struct.ClassProb** %_M_finish to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !37
  %2 = bitcast %"class.std::vector.3"* %this to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !40
  %sub.ptr.sub = sub i64 %1, %3
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 3
  ret i64 %sub.ptr.div
}

; Function Attrs: noreturn
declare dso_local void @_ZSt20__throw_length_errorPKc(i8*) local_unnamed_addr #16

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) i64* @_ZSt3maxImERKT_S2_S2_(i64* dereferenceable(8) %__a, i64* dereferenceable(8) %__b) local_unnamed_addr #12 comdat {
entry:
  %0 = load i64, i64* %__a, align 8, !tbaa !20
  %1 = load i64, i64* %__b, align 8, !tbaa !20
  %cmp = icmp ult i64 %0, %1
  %__b.__a = select i1 %cmp, i64* %__b, i64* %__a
  ret i64* %__b.__a
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt6vectorI9ClassProbSaIS0_EE11_S_max_sizeERKS1_(%"class.std::allocator.5"* dereferenceable(1) %__a) local_unnamed_addr #4 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %__diffmax = alloca i64, align 8
  %__allocmax = alloca i64, align 8
  %0 = bitcast i64* %__diffmax to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  store i64 1152921504606846975, i64* %__diffmax, align 8, !tbaa !20
  %1 = bitcast i64* %__allocmax to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #20
  %call = tail call i64 @_ZNSt16allocator_traitsISaI9ClassProbEE8max_sizeERKS1_(%"class.std::allocator.5"* nonnull dereferenceable(1) %__a) #20
  store i64 %call, i64* %__allocmax, align 8, !tbaa !20
  %call1 = call dereferenceable(8) i64* @_ZSt3minImERKT_S2_S2_(i64* nonnull dereferenceable(8) %__diffmax, i64* nonnull dereferenceable(8) %__allocmax)
  %2 = load i64, i64* %call1, align 8, !tbaa !20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  ret i64 %2
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(1) %"class.std::allocator.5"* @_ZNKSt12_Vector_baseI9ClassProbSaIS0_EE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base.4"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"struct.std::_Vector_base.4"* %this to %"class.std::allocator.5"*
  ret %"class.std::allocator.5"* %0
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt16allocator_traitsISaI9ClassProbEE8max_sizeERKS1_(%"class.std::allocator.5"* dereferenceable(1) %__a) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator.5"* %__a to %"class.__gnu_cxx::new_allocator.6"*
  %call = tail call i64 @_ZNK9__gnu_cxx13new_allocatorI9ClassProbE8max_sizeEv(%"class.__gnu_cxx::new_allocator.6"* nonnull %0) #20
  ret i64 %call
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) i64* @_ZSt3minImERKT_S2_S2_(i64* dereferenceable(8) %__a, i64* dereferenceable(8) %__b) local_unnamed_addr #12 comdat {
entry:
  %0 = load i64, i64* %__b, align 8, !tbaa !20
  %1 = load i64, i64* %__a, align 8, !tbaa !20
  %cmp = icmp ult i64 %0, %1
  %__b.__a = select i1 %cmp, i64* %__b, i64* %__a
  ret i64* %__b.__a
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK9__gnu_cxx13new_allocatorI9ClassProbE8max_sizeEv(%"class.__gnu_cxx::new_allocator.6"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  ret i64 1152921504606846975
}

; Function Attrs: uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZNSt16allocator_traitsISaI9ClassProbEE8allocateERS1_m(%"class.std::allocator.5"* dereferenceable(1) %__a, i64 %__n) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator.5"* %__a to %"class.__gnu_cxx::new_allocator.6"*
  %call = tail call %struct.ClassProb* @_ZN9__gnu_cxx13new_allocatorI9ClassProbE8allocateEmPKv(%"class.__gnu_cxx::new_allocator.6"* nonnull %0, i64 %__n, i8* null)
  ret %struct.ClassProb* %call
}

; Function Attrs: uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZN9__gnu_cxx13new_allocatorI9ClassProbE8allocateEmPKv(%"class.__gnu_cxx::new_allocator.6"* %this, i64 %__n, i8*) local_unnamed_addr #0 comdat align 2 {
entry:
  %call = tail call i64 @_ZNK9__gnu_cxx13new_allocatorI9ClassProbE8max_sizeEv(%"class.__gnu_cxx::new_allocator.6"* %this) #20
  %cmp = icmp ult i64 %call, %__n
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @_ZSt17__throw_bad_allocv() #22
  unreachable

if.end:                                           ; preds = %entry
  %mul = shl i64 %__n, 3
  %call2 = tail call i8* @_Znwm(i64 %mul)
  %1 = bitcast i8* %call2 to %struct.ClassProb*
  ret %struct.ClassProb* %1
}

; Function Attrs: noreturn
declare dso_local void @_ZSt17__throw_bad_allocv() local_unnamed_addr #16

; Function Attrs: nobuiltin nofree
declare dso_local noalias nonnull i8* @_Znwm(i64) local_unnamed_addr #18

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZNSt6vectorI9ClassProbSaIS0_EE14_S_do_relocateEPS0_S3_S3_RS1_St17integral_constantIbLb1EE(%struct.ClassProb* %__first, %struct.ClassProb* %__last, %struct.ClassProb* %__result, %"class.std::allocator.5"* dereferenceable(1) %__alloc) local_unnamed_addr #4 comdat align 2 {
entry:
  %call = tail call %struct.ClassProb* @_ZSt12__relocate_aIP9ClassProbS1_SaIS0_EET0_T_S4_S3_RT1_(%struct.ClassProb* %__first, %struct.ClassProb* %__last, %struct.ClassProb* %__result, %"class.std::allocator.5"* nonnull dereferenceable(1) %__alloc) #20
  ret %struct.ClassProb* %call
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt12__relocate_aIP9ClassProbS1_SaIS0_EET0_T_S4_S3_RT1_(%struct.ClassProb* %__first, %struct.ClassProb* %__last, %struct.ClassProb* %__result, %"class.std::allocator.5"* dereferenceable(1) %__alloc) local_unnamed_addr #12 comdat {
entry:
  %call = tail call %struct.ClassProb* @_ZSt12__niter_baseIP9ClassProbET_S2_(%struct.ClassProb* %__first) #20
  %call1 = tail call %struct.ClassProb* @_ZSt12__niter_baseIP9ClassProbET_S2_(%struct.ClassProb* %__last) #20
  %call2 = tail call %struct.ClassProb* @_ZSt12__niter_baseIP9ClassProbET_S2_(%struct.ClassProb* %__result) #20
  %call3 = tail call %struct.ClassProb* @_ZSt14__relocate_a_1I9ClassProbS0_ENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS2_E4typeES3_S3_S3_RSaIT0_E(%struct.ClassProb* %call, %struct.ClassProb* %call1, %struct.ClassProb* %call2, %"class.std::allocator.5"* nonnull dereferenceable(1) %__alloc) #20
  ret %struct.ClassProb* %call3
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt14__relocate_a_1I9ClassProbS0_ENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS2_E4typeES3_S3_S3_RSaIT0_E(%struct.ClassProb* %__first, %struct.ClassProb* %__last, %struct.ClassProb* %__result, %"class.std::allocator.5"* dereferenceable(1)) local_unnamed_addr #12 comdat {
entry:
  %sub.ptr.lhs.cast = ptrtoint %struct.ClassProb* %__last to i64
  %sub.ptr.rhs.cast = ptrtoint %struct.ClassProb* %__first to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 3
  %cmp = icmp sgt i64 %sub.ptr.sub, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = bitcast %struct.ClassProb* %__result to i8*
  %2 = bitcast %struct.ClassProb* %__first to i8*
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* align 4 %1, i8* align 4 %2, i64 %sub.ptr.sub, i1 false)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %add.ptr = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %__result, i64 %sub.ptr.div
  ret %struct.ClassProb* %add.ptr
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt12__niter_baseIP9ClassProbET_S2_(%struct.ClassProb* %__it) local_unnamed_addr #12 comdat {
entry:
  ret %struct.ClassProb* %__it
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1 immarg) #5

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEC2ERKS2_(%"class.__gnu_cxx::__normal_iterator"* %this, %struct.ClassProb** dereferenceable(8) %__i) unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %struct.ClassProb** %__i to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %2 = bitcast %"class.__gnu_cxx::__normal_iterator"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !75
  ret void
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZSt6__sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #10 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %call = call zeroext i1 @_ZN9__gnu_cxxneIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last) #20
  br i1 %call, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %agg.tmp3.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  %call4 = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first) #20
  %call5 = call i64 @_ZSt4__lgl(i64 %call4)
  %mul = shl nsw i64 %call5, 1
  call void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElNS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_T1_(%struct.ClassProb* %agg.tmp.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp3.sroa.0.0.copyload, i64 %mul, i1 (i64, i64)* %__comp.coerce)
  %agg.tmp10.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %agg.tmp11.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  call void @_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_(%struct.ClassProb* %agg.tmp10.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp11.sroa.0.0.copyload, i1 (i64, i64)* %__comp.coerce)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local i1 (i64, i64)* @_ZN9__gnu_cxx5__ops16__iter_comp_iterIPFb9ClassProbS2_EEENS0_15_Iter_comp_iterIT_EES6_(i1 (i64, i64)* %__comp) local_unnamed_addr #10 comdat {
entry:
  %retval = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter", align 8
  %__comp.addr = alloca i1 (i64, i64)*, align 8
  store i1 (i64, i64)* %__comp, i1 (i64, i64)** %__comp.addr, align 8, !tbaa !58
  %call = call dereferenceable(8) i1 (i64, i64)** @_ZSt4moveIRPFb9ClassProbS0_EEONSt16remove_referenceIT_E4typeEOS5_(i1 (i64, i64)** nonnull dereferenceable(8) %__comp.addr) #20
  %0 = load i1 (i64, i64)*, i1 (i64, i64)** %call, align 8, !tbaa !58
  call void @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEC2ES4_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %retval, i1 (i64, i64)* %0)
  %coerce.dive = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %retval, i64 0, i32 0
  %1 = load i1 (i64, i64)*, i1 (i64, i64)** %coerce.dive, align 8
  ret i1 (i64, i64)* %1
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local zeroext i1 @_ZN9__gnu_cxxneIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* dereferenceable(8) %__lhs, %"class.__gnu_cxx::__normal_iterator"* dereferenceable(8) %__rhs) local_unnamed_addr #12 comdat {
entry:
  %call = tail call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__lhs) #20
  %0 = load %struct.ClassProb*, %struct.ClassProb** %call, align 8, !tbaa !58
  %call1 = tail call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__rhs) #20
  %1 = load %struct.ClassProb*, %struct.ClassProb** %call1, align 8, !tbaa !58
  %cmp = icmp ne %struct.ClassProb* %0, %1
  ret i1 %cmp
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElNS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_T1_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, i64 %__depth_limit, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #0 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %call28 = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first) #20
  %cmp29 = icmp sgt i64 %call28, 16
  br i1 %cmp29, label %while.body, label %while.end

while.body:                                       ; preds = %if.end, %entry
  %__depth_limit.addr.030 = phi i64 [ %dec, %if.end ], [ %__depth_limit, %entry ]
  %cmp3 = icmp eq i64 %__depth_limit.addr.030, 0
  br i1 %cmp3, label %if.then, label %if.end

if.then:                                          ; preds = %while.body
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %agg.tmp4.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  call void @_ZSt14__partial_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_T0_(%struct.ClassProb* %agg.tmp.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp4.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp4.sroa.0.0.copyload, i1 (i64, i64)* %__comp.coerce)
  br label %while.end

if.end:                                           ; preds = %while.body
  %dec = add nsw i64 %__depth_limit.addr.030, -1
  %agg.tmp11.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %agg.tmp12.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  %call17 = call %struct.ClassProb* @_ZSt27__unguarded_partition_pivotIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEET_SD_SD_T0_(%struct.ClassProb* %agg.tmp11.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp12.sroa.0.0.copyload, i1 (i64, i64)* %__comp.coerce)
  %agg.tmp20.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  call void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElNS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_T1_(%struct.ClassProb* %call17, %struct.ClassProb* %agg.tmp20.sroa.0.0.copyload, i64 %dec, i1 (i64, i64)* %__comp.coerce)
  store %struct.ClassProb* %call17, %struct.ClassProb** %coerce.dive1, align 8
  %call = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first) #20
  %cmp = icmp sgt i64 %call, 16
  br i1 %cmp, label %while.body, label %while.end

while.end:                                        ; preds = %if.end, %if.then, %entry
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i64 @_ZSt4__lgl(i64 %__n) local_unnamed_addr #12 comdat {
entry:
  %0 = tail call i64 @llvm.ctlz.i64(i64 %__n, i1 true), !range !77
  %sub = xor i64 %0, 63
  ret i64 %sub
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZSt22__final_insertion_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #0 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %call = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first) #20
  %cmp = icmp sgt i64 %call, 16
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %call4 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 16) #20
  call void @_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_(%struct.ClassProb* %agg.tmp.sroa.0.0.copyload, %struct.ClassProb* %call4, i1 (i64, i64)* %__comp.coerce)
  %call11 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 16) #20
  %agg.tmp13.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  call void @_ZSt26__unguarded_insertion_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_(%struct.ClassProb* %call11, %struct.ClassProb* %agg.tmp13.sroa.0.0.copyload, i1 (i64, i64)* %__comp.coerce)
  br label %if.end

if.else:                                          ; preds = %entry
  %agg.tmp19.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  call void @_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_(%struct.ClassProb* %agg.tmp.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp19.sroa.0.0.copyload, i1 (i64, i64)* %__comp.coerce)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZSt14__partial_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__middle.coerce, %struct.ClassProb* %__last.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #10 comdat {
entry:
  %__comp = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter", align 8
  %coerce.dive3 = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  store i1 (i64, i64)* %__comp.coerce, i1 (i64, i64)** %coerce.dive3, align 8
  tail call void @_ZSt13__heap_selectIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__middle.coerce, %struct.ClassProb* %__last.coerce, i1 (i64, i64)* %__comp.coerce)
  call void @_ZSt11__sort_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_RT0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__middle.coerce, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull dereferenceable(8) %__comp)
  ret void
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt27__unguarded_partition_pivotIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEET_SD_SD_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #10 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %call = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first) #20
  %div = sdiv i64 %call, 2
  %call3 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %div) #20
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %call6 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 1) #20
  %call10 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEmiEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__last, i64 1) #20
  call void @_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_SD_T0_(%struct.ClassProb* %agg.tmp.sroa.0.0.copyload, %struct.ClassProb* %call6, %struct.ClassProb* %call3, %struct.ClassProb* %call10, i1 (i64, i64)* %__comp.coerce)
  %call19 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 1) #20
  %agg.tmp21.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  %agg.tmp22.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %call28 = call %struct.ClassProb* @_ZSt21__unguarded_partitionIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEET_SD_SD_SD_T0_(%struct.ClassProb* %call19, %struct.ClassProb* %agg.tmp21.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp22.sroa.0.0.copyload, i1 (i64, i64)* %__comp.coerce)
  ret %struct.ClassProb* %call28
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZSt13__heap_selectIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__middle.coerce, %struct.ClassProb* %__last.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #0 comdat {
entry:
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__comp = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter", align 8
  %__i = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive2 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive2, align 8
  %coerce.dive3 = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  store i1 (i64, i64)* %__comp.coerce, i1 (i64, i64)** %coerce.dive3, align 8
  call void @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_RT0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__middle.coerce, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull dereferenceable(8) %__comp)
  %0 = bitcast %"class.__gnu_cxx::__normal_iterator"* %__i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %__middle.sroa.0.0..sroa_idx = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__i, i64 0, i32 0
  store %struct.ClassProb* %__middle.coerce, %struct.ClassProb** %__middle.sroa.0.0..sroa_idx, align 8
  %call19 = call zeroext i1 @_ZN9__gnu_cxxltIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__i, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last) #20
  br i1 %call19, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  ret void

for.body:                                         ; preds = %for.inc, %entry
  %agg.tmp7.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %__middle.sroa.0.0..sroa_idx, align 8
  %call11 = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %agg.tmp7.sroa.0.0.copyload, %struct.ClassProb* %__first.coerce)
  br i1 %call11, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %agg.tmp14.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %__middle.sroa.0.0..sroa_idx, align 8
  call void @_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_RT0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__middle.coerce, %struct.ClassProb* %agg.tmp14.sroa.0.0.copyload, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull dereferenceable(8) %__comp)
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %call18 = call dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEppEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__i) #20
  %call = call zeroext i1 @_ZN9__gnu_cxxltIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__i, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last) #20
  br i1 %call, label %for.body, label %for.cond.cleanup
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZSt11__sort_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_RT0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* dereferenceable(8) %__comp) local_unnamed_addr #0 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %call8 = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first) #20
  %cmp9 = icmp sgt i64 %call8, 1
  br i1 %cmp9, label %while.body, label %while.end

while.body:                                       ; preds = %while.body, %entry
  %call2 = call dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEmmEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__last) #20
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %agg.tmp3.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  call void @_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_RT0_(%struct.ClassProb* %agg.tmp.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp3.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp3.sroa.0.0.copyload, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull dereferenceable(8) %__comp)
  %call = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first) #20
  %cmp = icmp sgt i64 %call, 1
  br i1 %cmp, label %while.body, label %while.end

while.end:                                        ; preds = %while.body, %entry
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_RT0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* dereferenceable(8) %__comp) local_unnamed_addr #0 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__value = alloca i64, align 8
  %tmpcast = bitcast i64* %__value to %struct.ClassProb*
  %ref.tmp = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %call = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first) #20
  %cmp = icmp slt i64 %call, 2
  br i1 %cmp, label %return, label %if.end

if.end:                                           ; preds = %entry
  %call2 = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first) #20
  %sub = add nsw i64 %call2, -2
  %div = sdiv i64 %sub, 2
  %0 = bitcast i64* %__value to i8*
  %1 = bitcast %"class.__gnu_cxx::__normal_iterator"* %ref.tmp to i8*
  %coerce.dive4 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %ref.tmp, i64 0, i32 0
  %agg.tmp9.sroa.0.0..sroa_idx = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  br label %while.cond

while.cond:                                       ; preds = %while.cond, %if.end
  %__parent.0 = phi i64 [ %div, %if.end ], [ %dec, %while.cond ]
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #20
  %call3 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %__parent.0) #20
  store %struct.ClassProb* %call3, %struct.ClassProb** %coerce.dive4, align 8
  %call5 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %ref.tmp) #20
  %call6 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %call5) #20
  %2 = bitcast %struct.ClassProb* %call6 to i64*
  %3 = load i64, i64* %2, align 4
  store i64 %3, i64* %__value, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #20
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %call8 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %tmpcast) #20
  %agg.tmp7.sroa.0.0..sroa_cast = bitcast %struct.ClassProb* %call8 to i64*
  %agg.tmp7.sroa.0.0.copyload = load i64, i64* %agg.tmp7.sroa.0.0..sroa_cast, align 4
  %agg.tmp9.sroa.0.0.copyload = load i1 (i64, i64)*, i1 (i64, i64)** %agg.tmp9.sroa.0.0..sroa_idx, align 8
  call void @_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElS2_NS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_T0_SE_T1_T2_(%struct.ClassProb* %agg.tmp.sroa.0.0.copyload, i64 %__parent.0, i64 %call2, i64 %agg.tmp7.sroa.0.0.copyload, i1 (i64, i64)* %agg.tmp9.sroa.0.0.copyload)
  %cmp12 = icmp eq i64 %__parent.0, 0
  %dec = add nsw i64 %__parent.0, -1
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  br i1 %cmp12, label %return, label %while.cond

return:                                           ; preds = %while.cond, %entry
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local zeroext i1 @_ZN9__gnu_cxxltIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* dereferenceable(8) %__lhs, %"class.__gnu_cxx::__normal_iterator"* dereferenceable(8) %__rhs) local_unnamed_addr #12 comdat {
entry:
  %call = tail call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__lhs) #20
  %0 = load %struct.ClassProb*, %struct.ClassProb** %call, align 8, !tbaa !58
  %call1 = tail call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__rhs) #20
  %1 = load %struct.ClassProb*, %struct.ClassProb** %call1, align 8, !tbaa !58
  %cmp = icmp ult %struct.ClassProb* %0, %1
  ret i1 %cmp
}

; Function Attrs: uwtable
define linkonce_odr dso_local zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %this, %struct.ClassProb* %__it1.coerce, %struct.ClassProb* %__it2.coerce) local_unnamed_addr #0 comdat align 2 {
entry:
  %__it1 = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__it2 = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__it1, i64 0, i32 0
  store %struct.ClassProb* %__it1.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__it2, i64 0, i32 0
  store %struct.ClassProb* %__it2.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %_M_comp = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %this, i64 0, i32 0
  %0 = load i1 (i64, i64)*, i1 (i64, i64)** %_M_comp, align 8, !tbaa !78
  %call = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__it1) #20
  %agg.tmp.sroa.0.0..sroa_cast = bitcast %struct.ClassProb* %call to i64*
  %agg.tmp.sroa.0.0.copyload = load i64, i64* %agg.tmp.sroa.0.0..sroa_cast, align 4
  %call4 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__it2) #20
  %agg.tmp3.sroa.0.0..sroa_cast = bitcast %struct.ClassProb* %call4 to i64*
  %agg.tmp3.sroa.0.0.copyload = load i64, i64* %agg.tmp3.sroa.0.0..sroa_cast, align 4
  %call5 = call zeroext i1 %0(i64 %agg.tmp.sroa.0.0.copyload, i64 %agg.tmp3.sroa.0.0.copyload)
  ret i1 %call5
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_RT0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, %struct.ClassProb* %__result.coerce, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* dereferenceable(8) %__comp) local_unnamed_addr #10 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__result = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__value = alloca i64, align 8
  %tmpcast = bitcast i64* %__value to %struct.ClassProb*
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %coerce.dive2 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__result, i64 0, i32 0
  store %struct.ClassProb* %__result.coerce, %struct.ClassProb** %coerce.dive2, align 8
  %0 = bitcast i64* %__value to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__result) #20
  %call3 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %call) #20
  %1 = bitcast %struct.ClassProb* %call3 to i64*
  %2 = load i64, i64* %1, align 4
  store i64 %2, i64* %__value, align 8
  %call4 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first) #20
  %call5 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %call4) #20
  %call6 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__result) #20
  %3 = bitcast %struct.ClassProb* %call5 to i64*
  %4 = bitcast %struct.ClassProb* %call6 to i64*
  %5 = load i64, i64* %3, align 4
  store i64 %5, i64* %4, align 4
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %call7 = call i64 @_ZN9__gnu_cxxmiIP9ClassProbSt6vectorIS1_SaIS1_EEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS9_SC_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first) #20
  %call9 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %tmpcast) #20
  %agg.tmp8.sroa.0.0..sroa_cast = bitcast %struct.ClassProb* %call9 to i64*
  %agg.tmp8.sroa.0.0.copyload = load i64, i64* %agg.tmp8.sroa.0.0..sroa_cast, align 4
  %agg.tmp10.sroa.0.0..sroa_idx = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  %agg.tmp10.sroa.0.0.copyload = load i1 (i64, i64)*, i1 (i64, i64)** %agg.tmp10.sroa.0.0..sroa_idx, align 8
  call void @_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElS2_NS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_T0_SE_T1_T2_(%struct.ClassProb* %agg.tmp.sroa.0.0.copyload, i64 0, i64 %call7, i64 %agg.tmp8.sroa.0.0.copyload, i1 (i64, i64)* %agg.tmp10.sroa.0.0.copyload)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEppEv(%"class.__gnu_cxx::__normal_iterator"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_current = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %this, i64 0, i32 0
  %0 = load %struct.ClassProb*, %struct.ClassProb** %_M_current, align 8, !tbaa !75
  %incdec.ptr = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %0, i64 1
  store %struct.ClassProb* %incdec.ptr, %struct.ClassProb** %_M_current, align 8, !tbaa !75
  ret %"class.__gnu_cxx::__normal_iterator"* %this
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* dereferenceable(8) %__t) local_unnamed_addr #4 comdat {
entry:
  ret %struct.ClassProb* %__t
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* %this, i64 %__n) local_unnamed_addr #4 comdat align 2 {
entry:
  %retval = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %ref.tmp = alloca %struct.ClassProb*, align 8
  %0 = bitcast %struct.ClassProb** %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %_M_current = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %this, i64 0, i32 0
  %1 = load %struct.ClassProb*, %struct.ClassProb** %_M_current, align 8, !tbaa !75
  %add.ptr = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %1, i64 %__n
  store %struct.ClassProb* %add.ptr, %struct.ClassProb** %ref.tmp, align 8, !tbaa !58
  call void @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEC2ERKS2_(%"class.__gnu_cxx::__normal_iterator"* nonnull %retval, %struct.ClassProb** nonnull dereferenceable(8) %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %retval, i64 0, i32 0
  %2 = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  ret %struct.ClassProb* %2
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_current = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %this, i64 0, i32 0
  %0 = load %struct.ClassProb*, %struct.ClassProb** %_M_current, align 8, !tbaa !75
  ret %struct.ClassProb* %0
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElS2_NS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_T0_SE_T1_T2_(%struct.ClassProb* %__first.coerce, i64 %__holeIndex, i64 %__len, i64 %__value.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #0 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__value = alloca i64, align 8
  %tmpcast = bitcast i64* %__value to %struct.ClassProb*
  %__comp = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter", align 8
  %ref.tmp = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %ref.tmp14 = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %ref.tmp25 = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %ref.tmp31 = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__cmp = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_val", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  store i64 %__value.coerce, i64* %__value, align 8
  %coerce.dive1 = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  store i1 (i64, i64)* %__comp.coerce, i1 (i64, i64)** %coerce.dive1, align 8
  %sub = add nsw i64 %__len, -1
  %div = sdiv i64 %sub, 2
  %cmp60 = icmp sgt i64 %div, %__holeIndex
  br i1 %cmp60, label %while.body.lr.ph, label %while.end

while.body.lr.ph:                                 ; preds = %entry
  %0 = bitcast %"class.__gnu_cxx::__normal_iterator"* %ref.tmp to i8*
  %coerce.dive11 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %ref.tmp, i64 0, i32 0
  %1 = bitcast %"class.__gnu_cxx::__normal_iterator"* %ref.tmp14 to i8*
  %coerce.dive16 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %ref.tmp14, i64 0, i32 0
  br label %while.body

while.body:                                       ; preds = %while.body, %while.body.lr.ph
  %__secondChild.061 = phi i64 [ %__holeIndex, %while.body.lr.ph ], [ %spec.select, %while.body ]
  %add = shl i64 %__secondChild.061, 1
  %mul = add i64 %add, 2
  %call = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %mul) #20
  %sub4 = or i64 %add, 1
  %call5 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %sub4) #20
  %call9 = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %call, %struct.ClassProb* %call5)
  %spec.select = select i1 %call9, i64 %sub4, i64 %mul
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call10 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %spec.select) #20
  store %struct.ClassProb* %call10, %struct.ClassProb** %coerce.dive11, align 8
  %call12 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %ref.tmp) #20
  %call13 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %call12) #20
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #20
  %call15 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %__secondChild.061) #20
  store %struct.ClassProb* %call15, %struct.ClassProb** %coerce.dive16, align 8
  %call17 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %ref.tmp14) #20
  %2 = bitcast %struct.ClassProb* %call13 to i64*
  %3 = bitcast %struct.ClassProb* %call17 to i64*
  %4 = load i64, i64* %2, align 4
  store i64 %4, i64* %3, align 4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  %cmp = icmp slt i64 %spec.select, %div
  br i1 %cmp, label %while.body, label %while.end

while.end:                                        ; preds = %while.body, %entry
  %__secondChild.0.lcssa = phi i64 [ %__holeIndex, %entry ], [ %spec.select, %while.body ]
  %and = and i64 %__len, 1
  %cmp18 = icmp eq i64 %and, 0
  br i1 %cmp18, label %land.lhs.true, label %if.end36

land.lhs.true:                                    ; preds = %while.end
  %sub19 = add nsw i64 %__len, -2
  %div20 = sdiv i64 %sub19, 2
  %cmp21 = icmp eq i64 %__secondChild.0.lcssa, %div20
  br i1 %cmp21, label %if.then22, label %if.end36

if.then22:                                        ; preds = %land.lhs.true
  %add23 = shl i64 %__secondChild.0.lcssa, 1
  %5 = bitcast %"class.__gnu_cxx::__normal_iterator"* %ref.tmp25 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5) #20
  %sub26 = or i64 %add23, 1
  %call27 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %sub26) #20
  %coerce.dive28 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %ref.tmp25, i64 0, i32 0
  store %struct.ClassProb* %call27, %struct.ClassProb** %coerce.dive28, align 8
  %call29 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %ref.tmp25) #20
  %call30 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %call29) #20
  %6 = bitcast %"class.__gnu_cxx::__normal_iterator"* %ref.tmp31 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6) #20
  %call32 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %__secondChild.0.lcssa) #20
  %coerce.dive33 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %ref.tmp31, i64 0, i32 0
  store %struct.ClassProb* %call32, %struct.ClassProb** %coerce.dive33, align 8
  %call34 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %ref.tmp31) #20
  %7 = bitcast %struct.ClassProb* %call30 to i64*
  %8 = bitcast %struct.ClassProb* %call34 to i64*
  %9 = load i64, i64* %7, align 4
  store i64 %9, i64* %8, align 4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5) #20
  br label %if.end36

if.end36:                                         ; preds = %if.then22, %land.lhs.true, %while.end
  %__holeIndex.addr.1 = phi i64 [ %sub26, %if.then22 ], [ %__secondChild.0.lcssa, %land.lhs.true ], [ %__secondChild.0.lcssa, %while.end ]
  %10 = bitcast %"struct.__gnu_cxx::__ops::_Iter_comp_val"* %__cmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %10) #20
  %call37 = call dereferenceable(8) %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* @_ZSt4moveIRN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS3_EEEEONSt16remove_referenceIT_E4typeEOS9_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull dereferenceable(8) %__comp) #20
  call void @_ZN9__gnu_cxx5__ops14_Iter_comp_valIPFb9ClassProbS2_EEC2EONS0_15_Iter_comp_iterIS4_EE(%"struct.__gnu_cxx::__ops::_Iter_comp_val"* nonnull %__cmp, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull dereferenceable(8) %call37)
  %agg.tmp38.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %call40 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %tmpcast) #20
  %agg.tmp39.sroa.0.0..sroa_cast = bitcast %struct.ClassProb* %call40 to i64*
  %agg.tmp39.sroa.0.0.copyload = load i64, i64* %agg.tmp39.sroa.0.0..sroa_cast, align 4
  call void @_ZSt11__push_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElS2_NS0_5__ops14_Iter_comp_valIPFbS2_S2_EEEEvT_T0_SE_T1_RT2_(%struct.ClassProb* %agg.tmp38.sroa.0.0.copyload, i64 %__holeIndex.addr.1, i64 %__holeIndex, i64 %agg.tmp39.sroa.0.0.copyload, %"struct.__gnu_cxx::__ops::_Iter_comp_val"* nonnull dereferenceable(8) %__cmp)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %10) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* @_ZSt4moveIRN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS3_EEEEONSt16remove_referenceIT_E4typeEOS9_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* dereferenceable(8) %__t) local_unnamed_addr #4 comdat {
entry:
  ret %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__t
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx5__ops14_Iter_comp_valIPFb9ClassProbS2_EEC2EONS0_15_Iter_comp_iterIS4_EE(%"struct.__gnu_cxx::__ops::_Iter_comp_val"* %this, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* dereferenceable(8) %__comp) unnamed_addr #4 comdat align 2 {
entry:
  %_M_comp2 = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  %call = tail call dereferenceable(8) i1 (i64, i64)** @_ZSt4moveIRPFb9ClassProbS0_EEONSt16remove_referenceIT_E4typeEOS5_(i1 (i64, i64)** nonnull dereferenceable(8) %_M_comp2) #20
  %0 = bitcast i1 (i64, i64)** %call to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %2 = bitcast %"struct.__gnu_cxx::__ops::_Iter_comp_val"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !80
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZSt11__push_heapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEElS2_NS0_5__ops14_Iter_comp_valIPFbS2_S2_EEEEvT_T0_SE_T1_RT2_(%struct.ClassProb* %__first.coerce, i64 %__holeIndex, i64 %__topIndex, i64 %__value.coerce, %"struct.__gnu_cxx::__ops::_Iter_comp_val"* dereferenceable(8) %__comp) local_unnamed_addr #0 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__value = alloca i64, align 8
  %tmpcast = bitcast i64* %__value to %struct.ClassProb*
  %ref.tmp = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %ref.tmp8 = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %ref.tmp15 = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  store i64 %__value.coerce, i64* %__value, align 8
  %cmp28 = icmp sgt i64 %__holeIndex, %__topIndex
  br i1 %cmp28, label %land.rhs.lr.ph, label %while.end

land.rhs.lr.ph:                                   ; preds = %entry
  %0 = bitcast %"class.__gnu_cxx::__normal_iterator"* %ref.tmp to i8*
  %coerce.dive5 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %ref.tmp, i64 0, i32 0
  %1 = bitcast %"class.__gnu_cxx::__normal_iterator"* %ref.tmp8 to i8*
  %coerce.dive10 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %ref.tmp8, i64 0, i32 0
  br label %land.rhs

land.rhs:                                         ; preds = %while.body, %land.rhs.lr.ph
  %__parent.030.in.in = phi i64 [ %__holeIndex, %land.rhs.lr.ph ], [ %__parent.030, %while.body ]
  %__parent.030.in = add nsw i64 %__parent.030.in.in, -1
  %__parent.030 = sdiv i64 %__parent.030.in, 2
  %call = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %__parent.030) #20
  %call3 = call zeroext i1 @_ZN9__gnu_cxx5__ops14_Iter_comp_valIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEES2_EEbT_RT0_(%"struct.__gnu_cxx::__ops::_Iter_comp_val"* nonnull %__comp, %struct.ClassProb* %call, %struct.ClassProb* nonnull dereferenceable(8) %tmpcast)
  br i1 %call3, label %while.body, label %while.end

while.body:                                       ; preds = %land.rhs
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call4 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %__parent.030) #20
  store %struct.ClassProb* %call4, %struct.ClassProb** %coerce.dive5, align 8
  %call6 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %ref.tmp) #20
  %call7 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %call6) #20
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #20
  %call9 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %__parent.030.in.in) #20
  store %struct.ClassProb* %call9, %struct.ClassProb** %coerce.dive10, align 8
  %call11 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %ref.tmp8) #20
  %2 = bitcast %struct.ClassProb* %call7 to i64*
  %3 = bitcast %struct.ClassProb* %call11 to i64*
  %4 = load i64, i64* %2, align 4
  store i64 %4, i64* %3, align 4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  %cmp = icmp sgt i64 %__parent.030, %__topIndex
  br i1 %cmp, label %land.rhs, label %while.end

while.end:                                        ; preds = %while.body, %land.rhs, %entry
  %__parent.0.in.in.lcssa = phi i64 [ %__holeIndex, %entry ], [ %__parent.030, %while.body ], [ %__parent.030.in.in, %land.rhs ]
  %call14 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %tmpcast) #20
  %5 = bitcast %"class.__gnu_cxx::__normal_iterator"* %ref.tmp15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5) #20
  %call16 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 %__parent.0.in.in.lcssa) #20
  %coerce.dive17 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %ref.tmp15, i64 0, i32 0
  store %struct.ClassProb* %call16, %struct.ClassProb** %coerce.dive17, align 8
  %call18 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %ref.tmp15) #20
  %6 = bitcast %struct.ClassProb* %call14 to i64*
  %7 = bitcast %struct.ClassProb* %call18 to i64*
  %8 = load i64, i64* %6, align 4
  store i64 %8, i64* %7, align 4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5) #20
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) i1 (i64, i64)** @_ZSt4moveIRPFb9ClassProbS0_EEONSt16remove_referenceIT_E4typeEOS5_(i1 (i64, i64)** dereferenceable(8) %__t) local_unnamed_addr #4 comdat {
entry:
  ret i1 (i64, i64)** %__t
}

; Function Attrs: uwtable
define linkonce_odr dso_local zeroext i1 @_ZN9__gnu_cxx5__ops14_Iter_comp_valIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEES2_EEbT_RT0_(%"struct.__gnu_cxx::__ops::_Iter_comp_val"* %this, %struct.ClassProb* %__it.coerce, %struct.ClassProb* dereferenceable(8) %__val) local_unnamed_addr #0 comdat align 2 {
entry:
  %__it = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__it, i64 0, i32 0
  store %struct.ClassProb* %__it.coerce, %struct.ClassProb** %coerce.dive, align 8
  %_M_comp = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_val", %"struct.__gnu_cxx::__ops::_Iter_comp_val"* %this, i64 0, i32 0
  %0 = load i1 (i64, i64)*, i1 (i64, i64)** %_M_comp, align 8, !tbaa !80
  %call = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__it) #20
  %agg.tmp.sroa.0.0..sroa_cast = bitcast %struct.ClassProb* %call to i64*
  %agg.tmp.sroa.0.0.copyload = load i64, i64* %agg.tmp.sroa.0.0..sroa_cast, align 4
  %agg.tmp2.sroa.0.0..sroa_cast = bitcast %struct.ClassProb* %__val to i64*
  %agg.tmp2.sroa.0.0.copyload = load i64, i64* %agg.tmp2.sroa.0.0..sroa_cast, align 4
  %call3 = call zeroext i1 %0(i64 %agg.tmp.sroa.0.0.copyload, i64 %agg.tmp2.sroa.0.0.copyload)
  ret i1 %call3
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEmmEv(%"class.__gnu_cxx::__normal_iterator"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_current = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %this, i64 0, i32 0
  %0 = load %struct.ClassProb*, %struct.ClassProb** %_M_current, align 8, !tbaa !75
  %incdec.ptr = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %0, i64 -1
  store %struct.ClassProb* %incdec.ptr, %struct.ClassProb** %_M_current, align 8, !tbaa !75
  ret %"class.__gnu_cxx::__normal_iterator"* %this
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_SD_SD_T0_(%struct.ClassProb* %__result.coerce, %struct.ClassProb* %__a.coerce, %struct.ClassProb* %__b.coerce, %struct.ClassProb* %__c.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #0 comdat {
entry:
  %__comp = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter", align 8
  %coerce.dive4 = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  store i1 (i64, i64)* %__comp.coerce, i1 (i64, i64)** %coerce.dive4, align 8
  %call = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %__a.coerce, %struct.ClassProb* %__b.coerce)
  br i1 %call, label %if.then, label %if.else34

if.then:                                          ; preds = %entry
  %call12 = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %__b.coerce, %struct.ClassProb* %__c.coerce)
  br i1 %call12, label %if.end63, label %if.else

if.else:                                          ; preds = %if.then
  %call22 = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %__a.coerce, %struct.ClassProb* %__c.coerce)
  %__c.coerce.__a.coerce = select i1 %call22, %struct.ClassProb* %__c.coerce, %struct.ClassProb* %__a.coerce
  br label %if.end63

if.else34:                                        ; preds = %entry
  %call39 = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %__a.coerce, %struct.ClassProb* %__c.coerce)
  br i1 %call39, label %if.end63, label %if.else45

if.else45:                                        ; preds = %if.else34
  %call50 = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %__b.coerce, %struct.ClassProb* %__c.coerce)
  %__c.coerce.__b.coerce = select i1 %call50, %struct.ClassProb* %__c.coerce, %struct.ClassProb* %__b.coerce
  br label %if.end63

if.end63:                                         ; preds = %if.else45, %if.else34, %if.else, %if.then
  %__a.coerce.sink = phi %struct.ClassProb* [ %__b.coerce, %if.then ], [ %__c.coerce.__a.coerce, %if.else ], [ %__a.coerce, %if.else34 ], [ %__c.coerce.__b.coerce, %if.else45 ]
  call void @_ZSt9iter_swapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES7_EvT_T0_(%struct.ClassProb* %__result.coerce, %struct.ClassProb* %__a.coerce.sink)
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEmiEl(%"class.__gnu_cxx::__normal_iterator"* %this, i64 %__n) local_unnamed_addr #4 comdat align 2 {
entry:
  %retval = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %ref.tmp = alloca %struct.ClassProb*, align 8
  %0 = bitcast %struct.ClassProb** %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %_M_current = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %this, i64 0, i32 0
  %1 = load %struct.ClassProb*, %struct.ClassProb** %_M_current, align 8, !tbaa !75
  %idx.neg = sub i64 0, %__n
  %add.ptr = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %1, i64 %idx.neg
  store %struct.ClassProb* %add.ptr, %struct.ClassProb** %ref.tmp, align 8, !tbaa !58
  call void @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEC2ERKS2_(%"class.__gnu_cxx::__normal_iterator"* nonnull %retval, %struct.ClassProb** nonnull dereferenceable(8) %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %retval, i64 0, i32 0
  %2 = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  ret %struct.ClassProb* %2
}

; Function Attrs: uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt21__unguarded_partitionIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEET_SD_SD_SD_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, %struct.ClassProb* %__pivot.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #0 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__comp = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %coerce.dive3 = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  store i1 (i64, i64)* %__comp.coerce, i1 (i64, i64)** %coerce.dive3, align 8
  br label %while.body

while.body:                                       ; preds = %if.end, %entry
  %agg.tmp.sroa.0.0.copyload27 = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %call28 = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %agg.tmp.sroa.0.0.copyload27, %struct.ClassProb* %__pivot.coerce)
  br i1 %call28, label %while.body8, label %while.end

while.body8:                                      ; preds = %while.body8, %while.body
  %call9 = call dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEppEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first) #20
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %call = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %agg.tmp.sroa.0.0.copyload, %struct.ClassProb* %__pivot.coerce)
  br i1 %call, label %while.body8, label %while.end

while.end:                                        ; preds = %while.body8, %while.body
  %call10 = call dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEmmEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__last) #20
  %agg.tmp13.sroa.0.0.copyload29 = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  %call1630 = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %__pivot.coerce, %struct.ClassProb* %agg.tmp13.sroa.0.0.copyload29)
  br i1 %call1630, label %while.body17, label %while.end19

while.body17:                                     ; preds = %while.body17, %while.end
  %call18 = call dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEmmEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__last) #20
  %agg.tmp13.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  %call16 = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %__pivot.coerce, %struct.ClassProb* %agg.tmp13.sroa.0.0.copyload)
  br i1 %call16, label %while.body17, label %while.end19

while.end19:                                      ; preds = %while.body17, %while.end
  %call20 = call zeroext i1 @_ZN9__gnu_cxxltIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last) #20
  %agg.tmp21.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  br i1 %call20, label %if.end, label %if.then

if.then:                                          ; preds = %while.end19
  ret %struct.ClassProb* %agg.tmp21.sroa.0.0.copyload

if.end:                                           ; preds = %while.end19
  %agg.tmp22.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive1, align 8
  call void @_ZSt9iter_swapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES7_EvT_T0_(%struct.ClassProb* %agg.tmp21.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp22.sroa.0.0.copyload)
  %call25 = call dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEppEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first) #20
  br label %while.body
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZSt9iter_swapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES7_EvT_T0_(%struct.ClassProb* %__a.coerce, %struct.ClassProb* %__b.coerce) local_unnamed_addr #12 comdat {
entry:
  %__a = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__b = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__a, i64 0, i32 0
  store %struct.ClassProb* %__a.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__b, i64 0, i32 0
  store %struct.ClassProb* %__b.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %call = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__a) #20
  %call2 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__b) #20
  call void @_ZSt4swapI9ClassProbENSt9enable_ifIXsr6__and_ISt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SD_(%struct.ClassProb* nonnull dereferenceable(8) %call, %struct.ClassProb* nonnull dereferenceable(8) %call2) #20
  ret void
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local void @_ZSt4swapI9ClassProbENSt9enable_ifIXsr6__and_ISt6__not_ISt15__is_tuple_likeIT_EESt21is_move_constructibleIS4_ESt18is_move_assignableIS4_EEE5valueEvE4typeERS4_SD_(%struct.ClassProb* dereferenceable(8) %__a, %struct.ClassProb* dereferenceable(8) %__b) local_unnamed_addr #12 comdat {
entry:
  %__tmp = alloca i64, align 8
  %tmpcast = bitcast i64* %__tmp to %struct.ClassProb*
  %0 = bitcast i64* %__tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call = tail call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %__a) #20
  %1 = bitcast %struct.ClassProb* %call to i64*
  %2 = load i64, i64* %1, align 4
  store i64 %2, i64* %__tmp, align 8
  %call1 = tail call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %__b) #20
  %3 = bitcast %struct.ClassProb* %call1 to i64*
  %4 = bitcast %struct.ClassProb* %__a to i64*
  %5 = load i64, i64* %3, align 4
  store i64 %5, i64* %4, align 4
  %call2 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %tmpcast) #20
  %6 = bitcast %struct.ClassProb* %call2 to i64*
  %7 = bitcast %struct.ClassProb* %__b to i64*
  %8 = load i64, i64* %6, align 4
  store i64 %8, i64* %7, align 4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare i64 @llvm.ctlz.i64(i64, i1 immarg) #19

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZSt16__insertion_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #0 comdat {
entry:
  %__first = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__comp = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter", align 8
  %__i = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__val = alloca i64, align 8
  %tmpcast = bitcast i64* %__val to %struct.ClassProb*
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__first, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %coerce.dive2 = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  store i1 (i64, i64)* %__comp.coerce, i1 (i64, i64)** %coerce.dive2, align 8
  %call = call zeroext i1 @_ZN9__gnu_cxxeqIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__first, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last) #20
  br i1 %call, label %for.end, label %if.end

if.end:                                           ; preds = %entry
  %0 = bitcast %"class.__gnu_cxx::__normal_iterator"* %__i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call3 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first, i64 1) #20
  %coerce.dive4 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__i, i64 0, i32 0
  store %struct.ClassProb* %call3, %struct.ClassProb** %coerce.dive4, align 8
  %call535 = call zeroext i1 @_ZN9__gnu_cxxneIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__i, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last) #20
  br i1 %call535, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %if.end
  %1 = bitcast i64* %__val to i8*
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %if.end
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  br label %for.end

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive4, align 8
  %agg.tmp6.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %call9 = call zeroext i1 @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEclINS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEESC_EEbT_T0_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull %__comp, %struct.ClassProb* %agg.tmp.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp6.sroa.0.0.copyload)
  br i1 %call9, label %if.then10, label %if.else

if.then10:                                        ; preds = %for.body
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #20
  %call11 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__i) #20
  %call12 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %call11) #20
  %2 = bitcast %struct.ClassProb* %call12 to i64*
  %3 = load i64, i64* %2, align 4
  store i64 %3, i64* %__val, align 8
  %agg.tmp13.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive, align 8
  %agg.tmp14.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive4, align 8
  %call16 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__i, i64 1) #20
  %call21 = call %struct.ClassProb* @_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES7_ET0_T_S9_S8_(%struct.ClassProb* %agg.tmp13.sroa.0.0.copyload, %struct.ClassProb* %agg.tmp14.sroa.0.0.copyload, %struct.ClassProb* %call16)
  %call23 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %tmpcast) #20
  %call24 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__first) #20
  %4 = bitcast %struct.ClassProb* %call23 to i64*
  %5 = bitcast %struct.ClassProb* %call24 to i64*
  %6 = load i64, i64* %4, align 4
  store i64 %6, i64* %5, align 4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #20
  br label %for.inc

if.else:                                          ; preds = %for.body
  %agg.tmp25.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %coerce.dive4, align 8
  %agg.tmp27.sroa.0.0.copyload = load i1 (i64, i64)*, i1 (i64, i64)** %coerce.dive2, align 8
  %call29 = call i1 (i64, i64)* @_ZN9__gnu_cxx5__ops15__val_comp_iterIPFb9ClassProbS2_EEENS0_14_Val_comp_iterIT_EENS0_15_Iter_comp_iterIS6_EE(i1 (i64, i64)* %agg.tmp27.sroa.0.0.copyload)
  call void @_ZSt25__unguarded_linear_insertIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops14_Val_comp_iterIPFbS2_S2_EEEEvT_T0_(%struct.ClassProb* %agg.tmp25.sroa.0.0.copyload, i1 (i64, i64)* %call29)
  br label %for.inc

for.inc:                                          ; preds = %if.else, %if.then10
  %call34 = call dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEppEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__i) #20
  %call5 = call zeroext i1 @_ZN9__gnu_cxxneIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__i, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last) #20
  br i1 %call5, label %for.body, label %for.cond.cleanup

for.end:                                          ; preds = %for.cond.cleanup, %entry
  ret void
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local void @_ZSt26__unguarded_insertion_sortIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops15_Iter_comp_iterIPFbS2_S2_EEEEvT_SD_T0_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #10 comdat {
entry:
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__i = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive1 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive1, align 8
  %0 = bitcast %"class.__gnu_cxx::__normal_iterator"* %__i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %__first.sroa.0.0..sroa_idx = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__i, i64 0, i32 0
  store %struct.ClassProb* %__first.coerce, %struct.ClassProb** %__first.sroa.0.0..sroa_idx, align 8
  %call11 = call zeroext i1 @_ZN9__gnu_cxxneIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__i, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last) #20
  br i1 %call11, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body, %entry
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  ret void

for.body:                                         ; preds = %for.body, %entry
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %__first.sroa.0.0..sroa_idx, align 8
  %call6 = call i1 (i64, i64)* @_ZN9__gnu_cxx5__ops15__val_comp_iterIPFb9ClassProbS2_EEENS0_14_Val_comp_iterIT_EENS0_15_Iter_comp_iterIS6_EE(i1 (i64, i64)* %__comp.coerce)
  call void @_ZSt25__unguarded_linear_insertIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops14_Val_comp_iterIPFbS2_S2_EEEEvT_T0_(%struct.ClassProb* %agg.tmp.sroa.0.0.copyload, i1 (i64, i64)* %call6)
  %call10 = call dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEppEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__i) #20
  %call = call zeroext i1 @_ZN9__gnu_cxxneIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__i, %"class.__gnu_cxx::__normal_iterator"* nonnull dereferenceable(8) %__last) #20
  br i1 %call, label %for.body, label %for.cond.cleanup
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local zeroext i1 @_ZN9__gnu_cxxeqIP9ClassProbSt6vectorIS1_SaIS1_EEEEbRKNS_17__normal_iteratorIT_T0_EESB_(%"class.__gnu_cxx::__normal_iterator"* dereferenceable(8) %__lhs, %"class.__gnu_cxx::__normal_iterator"* dereferenceable(8) %__rhs) local_unnamed_addr #12 comdat {
entry:
  %call = tail call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__lhs) #20
  %0 = load %struct.ClassProb*, %struct.ClassProb** %call, align 8, !tbaa !58
  %call1 = tail call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__rhs) #20
  %1 = load %struct.ClassProb*, %struct.ClassProb** %call1, align 8, !tbaa !58
  %cmp = icmp eq %struct.ClassProb* %0, %1
  ret i1 %cmp
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt13move_backwardIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES7_ET0_T_S9_S8_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, %struct.ClassProb* %__result.coerce) local_unnamed_addr #10 comdat {
entry:
  %call = tail call %struct.ClassProb* @_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEEET_S8_(%struct.ClassProb* %__first.coerce)
  %call9 = tail call %struct.ClassProb* @_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEEET_S8_(%struct.ClassProb* %__last.coerce)
  %call15 = tail call %struct.ClassProb* @_ZSt23__copy_move_backward_a2ILb1EN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES7_ET1_T0_S9_S8_(%struct.ClassProb* %call, %struct.ClassProb* %call9, %struct.ClassProb* %__result.coerce)
  ret %struct.ClassProb* %call15
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZSt25__unguarded_linear_insertIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEENS0_5__ops14_Val_comp_iterIPFbS2_S2_EEEEvT_T0_(%struct.ClassProb* %__last.coerce, i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #0 comdat {
entry:
  %__last = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %__comp = alloca %"struct.__gnu_cxx::__ops::_Val_comp_iter", align 8
  %__val = alloca i64, align 8
  %tmpcast = bitcast i64* %__val to %struct.ClassProb*
  %__next = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__last, i64 0, i32 0
  store %struct.ClassProb* %__last.coerce, %struct.ClassProb** %coerce.dive, align 8
  %coerce.dive1 = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Val_comp_iter", %"struct.__gnu_cxx::__ops::_Val_comp_iter"* %__comp, i64 0, i32 0
  store i1 (i64, i64)* %__comp.coerce, i1 (i64, i64)** %coerce.dive1, align 8
  %0 = bitcast i64* %__val to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__last) #20
  %call2 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %call) #20
  %1 = bitcast %struct.ClassProb* %call2 to i64*
  %2 = load i64, i64* %1, align 4
  store i64 %2, i64* %__val, align 8
  %3 = bitcast %"class.__gnu_cxx::__normal_iterator"* %__next to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3) #20
  %4 = bitcast %"class.__gnu_cxx::__normal_iterator"* %__last to i64*
  %5 = bitcast %"class.__gnu_cxx::__normal_iterator"* %__next to i64*
  %6 = load i64, i64* %4, align 8, !tbaa !58
  store i64 %6, i64* %5, align 8, !tbaa !58
  %call3 = call dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEmmEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__next) #20
  %agg.tmp.sroa.0.0..sroa_idx = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__next, i64 0, i32 0
  %agg.tmp.sroa.0.0.copyload12 = load %struct.ClassProb*, %struct.ClassProb** %agg.tmp.sroa.0.0..sroa_idx, align 8
  %call513 = call zeroext i1 @_ZN9__gnu_cxx5__ops14_Val_comp_iterIPFb9ClassProbS2_EEclIS2_NS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEEEEbRT_T0_(%"struct.__gnu_cxx::__ops::_Val_comp_iter"* nonnull %__comp, %struct.ClassProb* nonnull dereferenceable(8) %tmpcast, %struct.ClassProb* %agg.tmp.sroa.0.0.copyload12)
  br i1 %call513, label %while.body, label %while.end

while.body:                                       ; preds = %while.body, %entry
  %call6 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__next) #20
  %call7 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %call6) #20
  %call8 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__last) #20
  %7 = bitcast %struct.ClassProb* %call7 to i64*
  %8 = bitcast %struct.ClassProb* %call8 to i64*
  %9 = load i64, i64* %7, align 4
  store i64 %9, i64* %8, align 4
  %10 = load i64, i64* %5, align 8, !tbaa !58
  store i64 %10, i64* %4, align 8, !tbaa !58
  %call9 = call dereferenceable(8) %"class.__gnu_cxx::__normal_iterator"* @_ZN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEmmEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__next) #20
  %agg.tmp.sroa.0.0.copyload = load %struct.ClassProb*, %struct.ClassProb** %agg.tmp.sroa.0.0..sroa_idx, align 8
  %call5 = call zeroext i1 @_ZN9__gnu_cxx5__ops14_Val_comp_iterIPFb9ClassProbS2_EEclIS2_NS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEEEEbRT_T0_(%"struct.__gnu_cxx::__ops::_Val_comp_iter"* nonnull %__comp, %struct.ClassProb* nonnull dereferenceable(8) %tmpcast, %struct.ClassProb* %agg.tmp.sroa.0.0.copyload)
  br i1 %call5, label %while.body, label %while.end

while.end:                                        ; preds = %while.body, %entry
  %call10 = call dereferenceable(8) %struct.ClassProb* @_ZSt4moveIR9ClassProbEONSt16remove_referenceIT_E4typeEOS3_(%struct.ClassProb* nonnull dereferenceable(8) %tmpcast) #20
  %call11 = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__last) #20
  %11 = bitcast %struct.ClassProb* %call10 to i64*
  %12 = bitcast %struct.ClassProb* %call11 to i64*
  %13 = load i64, i64* %11, align 4
  store i64 %13, i64* %12, align 4
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  ret void
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local i1 (i64, i64)* @_ZN9__gnu_cxx5__ops15__val_comp_iterIPFb9ClassProbS2_EEENS0_14_Val_comp_iterIT_EENS0_15_Iter_comp_iterIS6_EE(i1 (i64, i64)* %__comp.coerce) local_unnamed_addr #10 comdat {
entry:
  %retval = alloca %"struct.__gnu_cxx::__ops::_Val_comp_iter", align 8
  %__comp = alloca %"struct.__gnu_cxx::__ops::_Iter_comp_iter", align 8
  %coerce.dive = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  store i1 (i64, i64)* %__comp.coerce, i1 (i64, i64)** %coerce.dive, align 8
  %call = call dereferenceable(8) %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* @_ZSt4moveIRN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS3_EEEEONSt16remove_referenceIT_E4typeEOS9_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull dereferenceable(8) %__comp) #20
  call void @_ZN9__gnu_cxx5__ops14_Val_comp_iterIPFb9ClassProbS2_EEC2EONS0_15_Iter_comp_iterIS4_EE(%"struct.__gnu_cxx::__ops::_Val_comp_iter"* nonnull %retval, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* nonnull dereferenceable(8) %call)
  %coerce.dive1 = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Val_comp_iter", %"struct.__gnu_cxx::__ops::_Val_comp_iter"* %retval, i64 0, i32 0
  %0 = load i1 (i64, i64)*, i1 (i64, i64)** %coerce.dive1, align 8
  ret i1 (i64, i64)* %0
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt23__copy_move_backward_a2ILb1EN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES7_ET1_T0_S9_S8_(%struct.ClassProb* %__first.coerce, %struct.ClassProb* %__last.coerce, %struct.ClassProb* %__result.coerce) local_unnamed_addr #10 comdat {
entry:
  %call = tail call %struct.ClassProb* @_ZSt12__niter_baseIP9ClassProbSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE(%struct.ClassProb* %__first.coerce) #20
  %call7 = tail call %struct.ClassProb* @_ZSt12__niter_baseIP9ClassProbSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE(%struct.ClassProb* %__last.coerce) #20
  %call10 = tail call %struct.ClassProb* @_ZSt12__niter_baseIP9ClassProbSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE(%struct.ClassProb* %__result.coerce) #20
  %call11 = tail call %struct.ClassProb* @_ZSt22__copy_move_backward_aILb1EP9ClassProbS1_ET1_T0_S3_S2_(%struct.ClassProb* %call, %struct.ClassProb* %call7, %struct.ClassProb* %call10)
  %call13 = tail call %struct.ClassProb* @_ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_(%struct.ClassProb* %__result.coerce, %struct.ClassProb* %call11)
  ret %struct.ClassProb* %call13
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEEET_S8_(%struct.ClassProb* %__it.coerce) local_unnamed_addr #12 comdat {
entry:
  ret %struct.ClassProb* %__it.coerce
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_(%struct.ClassProb* %__from.coerce, %struct.ClassProb* %__res) local_unnamed_addr #12 comdat {
entry:
  %__from = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__from, i64 0, i32 0
  store %struct.ClassProb* %__from.coerce, %struct.ClassProb** %coerce.dive, align 8
  %call = tail call %struct.ClassProb* @_ZSt12__niter_baseIP9ClassProbSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE(%struct.ClassProb* %__from.coerce) #20
  %sub.ptr.lhs.cast = ptrtoint %struct.ClassProb* %__res to i64
  %sub.ptr.rhs.cast = ptrtoint %struct.ClassProb* %call to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 3
  %call2 = call %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEplEl(%"class.__gnu_cxx::__normal_iterator"* nonnull %__from, i64 %sub.ptr.div) #20
  ret %struct.ClassProb* %call2
}

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt22__copy_move_backward_aILb1EP9ClassProbS1_ET1_T0_S3_S2_(%struct.ClassProb* %__first, %struct.ClassProb* %__last, %struct.ClassProb* %__result) local_unnamed_addr #10 comdat {
entry:
  %call = tail call %struct.ClassProb* @_ZNSt20__copy_move_backwardILb1ELb1ESt26random_access_iterator_tagE13__copy_move_bI9ClassProbEEPT_PKS4_S7_S5_(%struct.ClassProb* %__first, %struct.ClassProb* %__last, %struct.ClassProb* %__result)
  ret %struct.ClassProb* %call
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZSt12__niter_baseIP9ClassProbSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE(%struct.ClassProb* %__it.coerce) local_unnamed_addr #4 comdat {
entry:
  %__it = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__it, i64 0, i32 0
  store %struct.ClassProb* %__it.coerce, %struct.ClassProb** %coerce.dive, align 8
  %call = call dereferenceable(8) %struct.ClassProb** @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEE4baseEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__it) #20
  %0 = load %struct.ClassProb*, %struct.ClassProb** %call, align 8, !tbaa !58
  ret %struct.ClassProb* %0
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local %struct.ClassProb* @_ZNSt20__copy_move_backwardILb1ELb1ESt26random_access_iterator_tagE13__copy_move_bI9ClassProbEEPT_PKS4_S7_S5_(%struct.ClassProb* %__first, %struct.ClassProb* %__last, %struct.ClassProb* %__result) local_unnamed_addr #4 comdat align 2 {
entry:
  %sub.ptr.lhs.cast = ptrtoint %struct.ClassProb* %__last to i64
  %sub.ptr.rhs.cast = ptrtoint %struct.ClassProb* %__first to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 3
  %tobool = icmp eq i64 %sub.ptr.sub, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %idx.neg = sub nsw i64 0, %sub.ptr.div
  %add.ptr = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %__result, i64 %idx.neg
  %0 = bitcast %struct.ClassProb* %add.ptr to i8*
  %1 = bitcast %struct.ClassProb* %__first to i8*
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 %1, i64 %sub.ptr.sub, i1 false)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %idx.neg1 = sub nsw i64 0, %sub.ptr.div
  %add.ptr2 = getelementptr inbounds %struct.ClassProb, %struct.ClassProb* %__result, i64 %idx.neg1
  ret %struct.ClassProb* %add.ptr2
}

; Function Attrs: uwtable
define linkonce_odr dso_local zeroext i1 @_ZN9__gnu_cxx5__ops14_Val_comp_iterIPFb9ClassProbS2_EEclIS2_NS_17__normal_iteratorIPS2_St6vectorIS2_SaIS2_EEEEEEbRT_T0_(%"struct.__gnu_cxx::__ops::_Val_comp_iter"* %this, %struct.ClassProb* dereferenceable(8) %__val, %struct.ClassProb* %__it.coerce) local_unnamed_addr #0 comdat align 2 {
entry:
  %__it = alloca %"class.__gnu_cxx::__normal_iterator", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator", %"class.__gnu_cxx::__normal_iterator"* %__it, i64 0, i32 0
  store %struct.ClassProb* %__it.coerce, %struct.ClassProb** %coerce.dive, align 8
  %_M_comp = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Val_comp_iter", %"struct.__gnu_cxx::__ops::_Val_comp_iter"* %this, i64 0, i32 0
  %0 = load i1 (i64, i64)*, i1 (i64, i64)** %_M_comp, align 8, !tbaa !82
  %agg.tmp.sroa.0.0..sroa_cast = bitcast %struct.ClassProb* %__val to i64*
  %agg.tmp.sroa.0.0.copyload = load i64, i64* %agg.tmp.sroa.0.0..sroa_cast, align 4
  %call = call dereferenceable(8) %struct.ClassProb* @_ZNK9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEdeEv(%"class.__gnu_cxx::__normal_iterator"* nonnull %__it) #20
  %agg.tmp2.sroa.0.0..sroa_cast = bitcast %struct.ClassProb* %call to i64*
  %agg.tmp2.sroa.0.0.copyload = load i64, i64* %agg.tmp2.sroa.0.0..sroa_cast, align 4
  %call3 = call zeroext i1 %0(i64 %agg.tmp.sroa.0.0.copyload, i64 %agg.tmp2.sroa.0.0.copyload)
  ret i1 %call3
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx5__ops14_Val_comp_iterIPFb9ClassProbS2_EEC2EONS0_15_Iter_comp_iterIS4_EE(%"struct.__gnu_cxx::__ops::_Val_comp_iter"* %this, %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* dereferenceable(8) %__comp) unnamed_addr #4 comdat align 2 {
entry:
  %_M_comp2 = getelementptr inbounds %"struct.__gnu_cxx::__ops::_Iter_comp_iter", %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %__comp, i64 0, i32 0
  %call = tail call dereferenceable(8) i1 (i64, i64)** @_ZSt4moveIRPFb9ClassProbS0_EEONSt16remove_referenceIT_E4typeEOS5_(i1 (i64, i64)** nonnull dereferenceable(8) %_M_comp2) #20
  %0 = bitcast i1 (i64, i64)** %call to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %2 = bitcast %"struct.__gnu_cxx::__ops::_Val_comp_iter"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !82
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEC2ES4_(%"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %this, i1 (i64, i64)* %__comp) unnamed_addr #4 comdat align 2 {
entry:
  %__comp.addr = alloca i1 (i64, i64)*, align 8
  store i1 (i64, i64)* %__comp, i1 (i64, i64)** %__comp.addr, align 8, !tbaa !58
  %call = call dereferenceable(8) i1 (i64, i64)** @_ZSt4moveIRPFb9ClassProbS0_EEONSt16remove_referenceIT_E4typeEOS5_(i1 (i64, i64)** nonnull dereferenceable(8) %__comp.addr) #20
  %0 = bitcast i1 (i64, i64)** %call to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %2 = bitcast %"struct.__gnu_cxx::__ops::_Iter_comp_iter"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !78
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZNSt16allocator_traitsISaIfEE9constructIfJRKfEEEvRS0_PT_DpOT0_(%"class.std::allocator"* dereferenceable(1) %__a, float* %__p, float* dereferenceable(4) %__args) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator"* %__a to %"class.__gnu_cxx::new_allocator"*
  %call = tail call dereferenceable(4) float* @_ZSt7forwardIRKfEOT_RNSt16remove_referenceIS2_E4typeE(float* nonnull dereferenceable(4) %__args) #20
  tail call void @_ZN9__gnu_cxx13new_allocatorIfE9constructIfJRKfEEEvPT_DpOT0_(%"class.__gnu_cxx::new_allocator"* nonnull %0, float* %__p, float* nonnull dereferenceable(4) %call) #20
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local void @_ZNSt6vectorIfSaIfEE17_M_realloc_insertIJRKfEEEvN9__gnu_cxx17__normal_iteratorIPfS1_EEDpOT_(%"class.std::vector"* %this, float* %__position.coerce, float* dereferenceable(4) %__args) local_unnamed_addr #0 comdat align 2 {
entry:
  %__position = alloca %"class.__gnu_cxx::__normal_iterator.10", align 8
  %ref.tmp = alloca %"class.__gnu_cxx::__normal_iterator.10", align 8
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.10", %"class.__gnu_cxx::__normal_iterator.10"* %__position, i64 0, i32 0
  store float* %__position.coerce, float** %coerce.dive, align 8
  %call = tail call i64 @_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc(%"class.std::vector"* %this, i64 1, i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.49, i64 0, i64 0))
  %0 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0
  %_M_start = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 0
  %1 = load float*, float** %_M_start, align 8, !tbaa !6
  %_M_finish = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 1
  %2 = load float*, float** %_M_finish, align 8, !tbaa !11
  %3 = bitcast %"class.__gnu_cxx::__normal_iterator.10"* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3) #20
  %call3 = tail call float* @_ZNSt6vectorIfSaIfEE5beginEv(%"class.std::vector"* %this) #20
  %coerce.dive4 = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.10", %"class.__gnu_cxx::__normal_iterator.10"* %ref.tmp, i64 0, i32 0
  store float* %call3, float** %coerce.dive4, align 8
  %call5 = call i64 @_ZN9__gnu_cxxmiIPfSt6vectorIfSaIfEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_(%"class.__gnu_cxx::__normal_iterator.10"* nonnull dereferenceable(8) %__position, %"class.__gnu_cxx::__normal_iterator.10"* nonnull dereferenceable(8) %ref.tmp) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3) #20
  %call6 = call float* @_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm(%"struct.std::_Vector_base"* %0, i64 %call)
  %4 = bitcast %"class.std::vector"* %this to %"class.std::allocator"*
  %add.ptr = getelementptr inbounds float, float* %call6, i64 %call5
  %call8 = call dereferenceable(4) float* @_ZSt7forwardIRKfEOT_RNSt16remove_referenceIS2_E4typeE(float* nonnull dereferenceable(4) %__args) #20
  call void @_ZNSt16allocator_traitsISaIfEE9constructIfJRKfEEEvRS0_PT_DpOT0_(%"class.std::allocator"* dereferenceable(1) %4, float* %add.ptr, float* nonnull dereferenceable(4) %call8) #20
  %call9 = call dereferenceable(8) float** @_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.10"* nonnull %__position) #20
  %5 = load float*, float** %call9, align 8, !tbaa !58
  %call10 = call dereferenceable(1) %"class.std::allocator"* @_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base"* %0) #20
  %call11 = call float* @_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_(float* %1, float* %5, float* %call6, %"class.std::allocator"* nonnull dereferenceable(1) %call10) #20
  %incdec.ptr = getelementptr inbounds float, float* %call11, i64 1
  %call12 = call dereferenceable(8) float** @_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.10"* nonnull %__position) #20
  %6 = load float*, float** %call12, align 8, !tbaa !58
  %call13 = call dereferenceable(1) %"class.std::allocator"* @_ZNSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base"* %0) #20
  %call14 = call float* @_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_(float* %6, float* %2, float* nonnull %incdec.ptr, %"class.std::allocator"* nonnull dereferenceable(1) %call13) #20
  %_M_end_of_storage = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 2
  %7 = bitcast float** %_M_end_of_storage to i64*
  %8 = load i64, i64* %7, align 8, !tbaa !41
  %sub.ptr.rhs.cast = ptrtoint float* %1 to i64
  %sub.ptr.sub = sub i64 %8, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 2
  call void @_ZNSt12_Vector_baseIfSaIfEE13_M_deallocateEPfm(%"struct.std::_Vector_base"* %0, float* %1, i64 %sub.ptr.div)
  store float* %call6, float** %_M_start, align 8, !tbaa !6
  store float* %call14, float** %_M_finish, align 8, !tbaa !11
  %add.ptr20 = getelementptr inbounds float, float* %call6, i64 %call
  store float* %add.ptr20, float** %_M_end_of_storage, align 8, !tbaa !41
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local float* @_ZNSt6vectorIfSaIfEE3endEv(%"class.std::vector"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %retval = alloca %"class.__gnu_cxx::__normal_iterator.10", align 8
  %_M_finish = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 1
  call void @_ZN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEC2ERKS1_(%"class.__gnu_cxx::__normal_iterator.10"* nonnull %retval, float** nonnull dereferenceable(8) %_M_finish) #20
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.10", %"class.__gnu_cxx::__normal_iterator.10"* %retval, i64 0, i32 0
  %0 = load float*, float** %coerce.dive, align 8
  ret float* %0
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx13new_allocatorIfE9constructIfJRKfEEEvPT_DpOT0_(%"class.__gnu_cxx::new_allocator"* %this, float* %__p, float* dereferenceable(4) %__args) local_unnamed_addr #4 comdat align 2 {
entry:
  %call = tail call dereferenceable(4) float* @_ZSt7forwardIRKfEOT_RNSt16remove_referenceIS2_E4typeE(float* nonnull dereferenceable(4) %__args) #20
  %0 = bitcast float* %call to i32*
  %1 = load i32, i32* %0, align 4, !tbaa !25
  %2 = bitcast float* %__p to i32*
  store i32 %1, i32* %2, align 4, !tbaa !25
  ret void
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(4) float* @_ZSt7forwardIRKfEOT_RNSt16remove_referenceIS2_E4typeE(float* dereferenceable(4) %__t) local_unnamed_addr #4 comdat {
entry:
  ret float* %__t
}

; Function Attrs: uwtable
define linkonce_odr dso_local i64 @_ZNKSt6vectorIfSaIfEE12_M_check_lenEmPKc(%"class.std::vector"* %this, i64 %__n, i8* %__s) local_unnamed_addr #0 comdat align 2 {
entry:
  %__n.addr = alloca i64, align 8
  %ref.tmp = alloca i64, align 8
  store i64 %__n, i64* %__n.addr, align 8, !tbaa !20
  %call = tail call i64 @_ZNKSt6vectorIfSaIfEE8max_sizeEv(%"class.std::vector"* %this) #20
  %call2 = tail call i64 @_ZNKSt6vectorIfSaIfEE4sizeEv(%"class.std::vector"* %this) #20
  %sub = sub i64 %call, %call2
  %cmp = icmp ult i64 %sub, %__n
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @_ZSt20__throw_length_errorPKc(i8* %__s) #22
  unreachable

if.end:                                           ; preds = %entry
  %call3 = tail call i64 @_ZNKSt6vectorIfSaIfEE4sizeEv(%"class.std::vector"* %this) #20
  %0 = bitcast i64* %ref.tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  %call4 = tail call i64 @_ZNKSt6vectorIfSaIfEE4sizeEv(%"class.std::vector"* %this) #20
  store i64 %call4, i64* %ref.tmp, align 8, !tbaa !20
  %call5 = call dereferenceable(8) i64* @_ZSt3maxImERKT_S2_S2_(i64* nonnull dereferenceable(8) %ref.tmp, i64* nonnull dereferenceable(8) %__n.addr)
  %1 = load i64, i64* %call5, align 8, !tbaa !20
  %add = add i64 %1, %call3
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  %call6 = call i64 @_ZNKSt6vectorIfSaIfEE4sizeEv(%"class.std::vector"* %this) #20
  %cmp7 = icmp ult i64 %add, %call6
  br i1 %cmp7, label %cond.true, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %if.end
  %call8 = call i64 @_ZNKSt6vectorIfSaIfEE8max_sizeEv(%"class.std::vector"* %this) #20
  %cmp9 = icmp ugt i64 %add, %call8
  br i1 %cmp9, label %cond.true, label %cond.end

cond.true:                                        ; preds = %lor.lhs.false, %if.end
  %call10 = call i64 @_ZNKSt6vectorIfSaIfEE8max_sizeEv(%"class.std::vector"* %this) #20
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %lor.lhs.false
  %cond = phi i64 [ %call10, %cond.true ], [ %add, %lor.lhs.false ]
  ret i64 %cond
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i64 @_ZN9__gnu_cxxmiIPfSt6vectorIfSaIfEEEENS_17__normal_iteratorIT_T0_E15difference_typeERKS8_SB_(%"class.__gnu_cxx::__normal_iterator.10"* dereferenceable(8) %__lhs, %"class.__gnu_cxx::__normal_iterator.10"* dereferenceable(8) %__rhs) local_unnamed_addr #12 comdat {
entry:
  %call = tail call dereferenceable(8) float** @_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.10"* nonnull %__lhs) #20
  %0 = bitcast float** %call to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %call1 = tail call dereferenceable(8) float** @_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.10"* nonnull %__rhs) #20
  %2 = bitcast float** %call1 to i64*
  %3 = load i64, i64* %2, align 8, !tbaa !58
  %sub.ptr.sub = sub i64 %1, %3
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 2
  ret i64 %sub.ptr.div
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local float* @_ZNSt6vectorIfSaIfEE5beginEv(%"class.std::vector"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %retval = alloca %"class.__gnu_cxx::__normal_iterator.10", align 8
  %_M_start = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0, i32 0, i32 0, i32 0
  call void @_ZN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEC2ERKS1_(%"class.__gnu_cxx::__normal_iterator.10"* nonnull %retval, float** dereferenceable(8) %_M_start) #20
  %coerce.dive = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.10", %"class.__gnu_cxx::__normal_iterator.10"* %retval, i64 0, i32 0
  %0 = load float*, float** %coerce.dive, align 8
  ret float* %0
}

; Function Attrs: uwtable
define linkonce_odr dso_local float* @_ZNSt12_Vector_baseIfSaIfEE11_M_allocateEm(%"struct.std::_Vector_base"* %this, i64 %__n) local_unnamed_addr #0 comdat align 2 {
entry:
  %cmp = icmp eq i64 %__n, 0
  br i1 %cmp, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = bitcast %"struct.std::_Vector_base"* %this to %"class.std::allocator"*
  %call = tail call float* @_ZNSt16allocator_traitsISaIfEE8allocateERS0_m(%"class.std::allocator"* dereferenceable(1) %0, i64 %__n)
  br label %cond.end

cond.end:                                         ; preds = %cond.true, %entry
  %cond = phi float* [ %call, %cond.true ], [ null, %entry ]
  ret float* %cond
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local float* @_ZNSt6vectorIfSaIfEE11_S_relocateEPfS2_S2_RS0_(float* %__first, float* %__last, float* %__result, %"class.std::allocator"* dereferenceable(1) %__alloc) local_unnamed_addr #4 comdat align 2 {
entry:
  %call = tail call float* @_ZNSt6vectorIfSaIfEE14_S_do_relocateEPfS2_S2_RS0_St17integral_constantIbLb1EE(float* %__first, float* %__last, float* %__result, %"class.std::allocator"* nonnull dereferenceable(1) %__alloc) #20
  ret float* %call
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(8) float** @_ZNK9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEE4baseEv(%"class.__gnu_cxx::__normal_iterator.10"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %_M_current = getelementptr inbounds %"class.__gnu_cxx::__normal_iterator.10", %"class.__gnu_cxx::__normal_iterator.10"* %this, i64 0, i32 0
  ret float** %_M_current
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNKSt6vectorIfSaIfEE8max_sizeEv(%"class.std::vector"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = getelementptr inbounds %"class.std::vector", %"class.std::vector"* %this, i64 0, i32 0
  %call = tail call dereferenceable(1) %"class.std::allocator"* @_ZNKSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base"* %0) #20
  %call2 = tail call i64 @_ZNSt6vectorIfSaIfEE11_S_max_sizeERKS0_(%"class.std::allocator"* nonnull dereferenceable(1) %call) #20
  ret i64 %call2
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt6vectorIfSaIfEE11_S_max_sizeERKS0_(%"class.std::allocator"* dereferenceable(1) %__a) local_unnamed_addr #4 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
entry:
  %__diffmax = alloca i64, align 8
  %__allocmax = alloca i64, align 8
  %0 = bitcast i64* %__diffmax to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %0) #20
  store i64 2305843009213693951, i64* %__diffmax, align 8, !tbaa !20
  %1 = bitcast i64* %__allocmax to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %1) #20
  %call = tail call i64 @_ZNSt16allocator_traitsISaIfEE8max_sizeERKS0_(%"class.std::allocator"* nonnull dereferenceable(1) %__a) #20
  store i64 %call, i64* %__allocmax, align 8, !tbaa !20
  %call1 = call dereferenceable(8) i64* @_ZSt3minImERKT_S2_S2_(i64* nonnull dereferenceable(8) %__diffmax, i64* nonnull dereferenceable(8) %__allocmax)
  %2 = load i64, i64* %call1, align 8, !tbaa !20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %1) #20
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %0) #20
  ret i64 %2
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local dereferenceable(1) %"class.std::allocator"* @_ZNKSt12_Vector_baseIfSaIfEE19_M_get_Tp_allocatorEv(%"struct.std::_Vector_base"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"struct.std::_Vector_base"* %this to %"class.std::allocator"*
  ret %"class.std::allocator"* %0
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNSt16allocator_traitsISaIfEE8max_sizeERKS0_(%"class.std::allocator"* dereferenceable(1) %__a) local_unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator"* %__a to %"class.__gnu_cxx::new_allocator"*
  %call = tail call i64 @_ZNK9__gnu_cxx13new_allocatorIfE8max_sizeEv(%"class.__gnu_cxx::new_allocator"* nonnull %0) #20
  ret i64 %call
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local i64 @_ZNK9__gnu_cxx13new_allocatorIfE8max_sizeEv(%"class.__gnu_cxx::new_allocator"* %this) local_unnamed_addr #4 comdat align 2 {
entry:
  ret i64 2305843009213693951
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local void @_ZN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEC2ERKS1_(%"class.__gnu_cxx::__normal_iterator.10"* %this, float** dereferenceable(8) %__i) unnamed_addr #4 comdat align 2 {
entry:
  %0 = bitcast float** %__i to i64*
  %1 = load i64, i64* %0, align 8, !tbaa !58
  %2 = bitcast %"class.__gnu_cxx::__normal_iterator.10"* %this to i64*
  store i64 %1, i64* %2, align 8, !tbaa !84
  ret void
}

; Function Attrs: uwtable
define linkonce_odr dso_local float* @_ZNSt16allocator_traitsISaIfEE8allocateERS0_m(%"class.std::allocator"* dereferenceable(1) %__a, i64 %__n) local_unnamed_addr #0 comdat align 2 {
entry:
  %0 = bitcast %"class.std::allocator"* %__a to %"class.__gnu_cxx::new_allocator"*
  %call = tail call float* @_ZN9__gnu_cxx13new_allocatorIfE8allocateEmPKv(%"class.__gnu_cxx::new_allocator"* nonnull %0, i64 %__n, i8* null)
  ret float* %call
}

; Function Attrs: uwtable
define linkonce_odr dso_local float* @_ZN9__gnu_cxx13new_allocatorIfE8allocateEmPKv(%"class.__gnu_cxx::new_allocator"* %this, i64 %__n, i8*) local_unnamed_addr #0 comdat align 2 {
entry:
  %call = tail call i64 @_ZNK9__gnu_cxx13new_allocatorIfE8max_sizeEv(%"class.__gnu_cxx::new_allocator"* %this) #20
  %cmp = icmp ult i64 %call, %__n
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @_ZSt17__throw_bad_allocv() #22
  unreachable

if.end:                                           ; preds = %entry
  %mul = shl i64 %__n, 2
  %call2 = tail call i8* @_Znwm(i64 %mul)
  %1 = bitcast i8* %call2 to float*
  ret float* %1
}

; Function Attrs: nounwind uwtable
define linkonce_odr dso_local float* @_ZNSt6vectorIfSaIfEE14_S_do_relocateEPfS2_S2_RS0_St17integral_constantIbLb1EE(float* %__first, float* %__last, float* %__result, %"class.std::allocator"* dereferenceable(1) %__alloc) local_unnamed_addr #4 comdat align 2 {
entry:
  %call = tail call float* @_ZSt12__relocate_aIPfS0_SaIfEET0_T_S3_S2_RT1_(float* %__first, float* %__last, float* %__result, %"class.std::allocator"* nonnull dereferenceable(1) %__alloc) #20
  ret float* %call
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float* @_ZSt12__relocate_aIPfS0_SaIfEET0_T_S3_S2_RT1_(float* %__first, float* %__last, float* %__result, %"class.std::allocator"* dereferenceable(1) %__alloc) local_unnamed_addr #12 comdat {
entry:
  %call = tail call float* @_ZSt12__niter_baseIPfET_S1_(float* %__first) #20
  %call1 = tail call float* @_ZSt12__niter_baseIPfET_S1_(float* %__last) #20
  %call2 = tail call float* @_ZSt12__niter_baseIPfET_S1_(float* %__result) #20
  %call3 = tail call float* @_ZSt14__relocate_a_1IffENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS1_E4typeES2_S2_S2_RSaIT0_E(float* %call, float* %call1, float* %call2, %"class.std::allocator"* nonnull dereferenceable(1) %__alloc) #20
  ret float* %call3
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float* @_ZSt14__relocate_a_1IffENSt9enable_ifIXsr3std24__is_bitwise_relocatableIT_EE5valueEPS1_E4typeES2_S2_S2_RSaIT0_E(float* %__first, float* %__last, float* %__result, %"class.std::allocator"* dereferenceable(1)) local_unnamed_addr #12 comdat {
entry:
  %sub.ptr.lhs.cast = ptrtoint float* %__last to i64
  %sub.ptr.rhs.cast = ptrtoint float* %__first to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 2
  %cmp = icmp sgt i64 %sub.ptr.sub, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = bitcast float* %__result to i8*
  %2 = bitcast float* %__first to i8*
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* align 4 %1, i8* align 4 %2, i64 %sub.ptr.sub, i1 false)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %add.ptr = getelementptr inbounds float, float* %__result, i64 %sub.ptr.div
  ret float* %add.ptr
}

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local float* @_ZSt12__niter_baseIPfET_S1_(float* %__it) local_unnamed_addr #12 comdat {
entry:
  ret float* %__it
}

; Function Attrs: uwtable
define available_externally dso_local void @_ZNSt9basic_iosIcSt11char_traitsIcEE8setstateESt12_Ios_Iostate(%"class.std::basic_ios"* %this, i32 %__state) local_unnamed_addr #0 align 2 {
entry:
  %call = tail call i32 @_ZNKSt9basic_iosIcSt11char_traitsIcEE7rdstateEv(%"class.std::basic_ios"* %this)
  %call2 = tail call i32 @_ZStorSt12_Ios_IostateS_(i32 %call, i32 %__state)
  tail call void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(%"class.std::basic_ios"* %this, i32 %call2)
  ret void
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* dereferenceable(272), i8*, i64) local_unnamed_addr #1

declare dso_local void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(%"class.std::basic_ios"*, i32) local_unnamed_addr #1

; Function Attrs: inlinehint nounwind uwtable
define linkonce_odr dso_local i32 @_ZStorSt12_Ios_IostateS_(i32 %__a, i32 %__b) local_unnamed_addr #12 comdat {
entry:
  %or = or i32 %__b, %__a
  ret i32 %or
}

; Function Attrs: nounwind uwtable
define available_externally dso_local i32 @_ZNKSt9basic_iosIcSt11char_traitsIcEE7rdstateEv(%"class.std::basic_ios"* %this) local_unnamed_addr #4 align 2 {
entry:
  %_M_streambuf_state = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %this, i64 0, i32 0, i32 5
  %0 = load i32, i32* %_M_streambuf_state, align 8, !tbaa !86
  ret i32 %0
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertIPKvEERSoT_(%"class.std::basic_ostream"*, i8*) local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define available_externally dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZSt5flushIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_(%"class.std::basic_ostream"* dereferenceable(272) %__os) local_unnamed_addr #10 {
entry:
  %call = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %__os)
  ret %"class.std::basic_ostream"* %call
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"*, i8 signext) local_unnamed_addr #1

; Function Attrs: uwtable
define available_externally dso_local signext i8 @_ZNKSt9basic_iosIcSt11char_traitsIcEE5widenEc(%"class.std::basic_ios"* %this, i8 signext %__c) local_unnamed_addr #0 align 2 {
entry:
  %_M_ctype = getelementptr inbounds %"class.std::basic_ios", %"class.std::basic_ios"* %this, i64 0, i32 5
  %0 = load %"class.std::ctype"*, %"class.std::ctype"** %_M_ctype, align 8, !tbaa !87
  %call = tail call dereferenceable(576) %"class.std::ctype"* @_ZSt13__check_facetISt5ctypeIcEERKT_PS3_(%"class.std::ctype"* %0)
  %call2 = tail call signext i8 @_ZNKSt5ctypeIcE5widenEc(%"class.std::ctype"* nonnull %call, i8 signext %__c)
  ret i8 %call2
}

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"*) local_unnamed_addr #1

; Function Attrs: inlinehint uwtable
define linkonce_odr dso_local dereferenceable(576) %"class.std::ctype"* @_ZSt13__check_facetISt5ctypeIcEERKT_PS3_(%"class.std::ctype"* %__f) local_unnamed_addr #10 comdat {
entry:
  %tobool = icmp eq %"class.std::ctype"* %__f, null
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @_ZSt16__throw_bad_castv() #22
  unreachable

if.end:                                           ; preds = %entry
  ret %"class.std::ctype"* %__f
}

; Function Attrs: uwtable
define linkonce_odr dso_local signext i8 @_ZNKSt5ctypeIcE5widenEc(%"class.std::ctype"* %this, i8 signext %__c) local_unnamed_addr #0 comdat align 2 {
entry:
  %_M_widen_ok = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %this, i64 0, i32 8
  %0 = load i8, i8* %_M_widen_ok, align 8, !tbaa !88
  %tobool = icmp eq i8 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %idxprom = zext i8 %__c to i64
  %arrayidx = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %this, i64 0, i32 9, i64 %idxprom
  %1 = load i8, i8* %arrayidx, align 1, !tbaa !36
  br label %return

if.end:                                           ; preds = %entry
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* nonnull %this)
  %2 = bitcast %"class.std::ctype"* %this to i8 (%"class.std::ctype"*, i8)***
  %vtable = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %2, align 8, !tbaa !28
  %vfn = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable, i64 6
  %3 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn, align 8
  %call = tail call signext i8 %3(%"class.std::ctype"* nonnull %this, i8 signext %__c)
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i8 [ %1, %if.then ], [ %call, %if.end ]
  ret i8 %retval.0
}

; Function Attrs: noreturn
declare dso_local void @_ZSt16__throw_bad_castv() local_unnamed_addr #16

declare dso_local void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"*) local_unnamed_addr #1

declare dso_local dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo9_M_insertImEERSoT_(%"class.std::basic_ostream"*, i64) local_unnamed_addr #1

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_simple_graph.cpp() #0 section ".text.startup" {
entry:
  tail call fastcc void @__cxx_global_var_init()
  tail call fastcc void @__cxx_global_var_init.1()
  tail call fastcc void @__cxx_global_var_init.2()
  ret void
}

; Function Attrs: nofree nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare i64 @fwrite_unlocked(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare i32 @putchar(i32) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare i64 @fread_unlocked(i8* nocapture, i64, i64, %struct._IO_FILE* nocapture) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare i32 @fputc_unlocked(i32, %struct._IO_FILE* nocapture) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #5

; Function Attrs: nounwind
declare i8* @llvm.hpvm.tensor.add(i8*, i8*) #20

; Function Attrs: nounwind
declare i8* @llvm.hpvm.tensor.relu(i8*) #20

; Function Attrs: nounwind
declare i8* @llvm.hpvm.tensor.tanh(i8*) #20

; Function Attrs: nounwind
declare i8* @llvm.hpvm.createNode(i8*) #20

; Function Attrs: nounwind
declare void @llvm.hpvm.bind.input(i8*, i32, i32, i1) #20

; Function Attrs: nounwind
declare i8* @llvm.hpvm.createEdge(i8*, i8*, i1, i32, i32, i1) #20

; Function Attrs: nounwind
declare void @llvm.hpvm.bind.output(i8*, i32, i32, i1) #20

; Function Attrs: nounwind
declare i8* @llvm.hpvm.grad(i8*, i8*, i32) #20

; Function Attrs: nounwind
declare i8* @llvm.hpvm.launch(i8*, i8*, i1) #20

; Function Attrs: nounwind
declare void @llvm.hpvm.wait(i8*) #20

declare i8* @dereferencePtrToPtr(i8*)

declare i8* @tensorAddCPUPure(i8*, i8*)

declare i8* @tensorTanhCPUPure(i8*)

declare i8* @tensorReluCPUPure(i8*)

declare i8* @tensorElementWiseMultiplyCPU(i8*, i8*)

declare i8* @tensorAddDerivativeCPU(i8*, i8*, i32)

declare i8* @tensorReluDerivativeCPU(i8*)

declare i8* @tensorTanhDerivativeCPU(i8*)

declare i8* @tensorGemmCPU(i8*, i8*)

define i8* @GradFunction(i8*) {
entry:
  %ArgumentGEP = getelementptr i8, i8* %0, i32 0
  %Argument = call i8* @dereferencePtrToPtr(i8* %ArgumentGEP)
  %ArgumentGEP1 = getelementptr i8, i8* %0, i32 16
  %Argument2 = call i8* @dereferencePtrToPtr(i8* %ArgumentGEP1)
  %1 = call i8* @tensorAddCPUPure(i8* %Argument, i8* %Argument2)
  %2 = call i8* @tensorReluCPUPure(i8* %1)
  %3 = call i8* @tensorTanhCPUPure(i8* %2)
  %4 = call i8* @tensorAddDerivativeCPU(i8* %Argument, i8* %Argument2, i32 0)
  %5 = call i8* @tensorReluDerivativeCPU(i8* %1)
  %6 = call i8* @tensorElementWiseMultiplyCPU(i8* %4, i8* %5)
  %7 = call i8* @tensorTanhDerivativeCPU(i8* %2)
  %8 = call i8* @tensorElementWiseMultiplyCPU(i8* %6, i8* %7)
  ret i8* %8
}

declare void @llvm_hpvm_initApproxhpvmRt(i32)

declare void @llvm_hpvm_cleanupApproxhpvmRt()

declare void @llvm_hpvm_initializeRuntimeController(i8*)

declare void @llvm_hpvm_clearRuntimeController()

declare i8* @wrapper_tensorAdd(i8*, i8*, i8*)

declare i8* @wrapper_tensorRelu(i8*, i8*)

declare i8* @wrapper_tensorTanh(i8*, i8*)

declare i8* @llvm_hpvm_cpu_launch(i8* (i8*)*, i8*)

declare void @llvm_hpvm_cpu_wait(i8*)

declare i8* @llvm_hpvm_cpu_argument_ptr(i8*, i64)

declare i8* @llvm_hpvm_streamLaunch(void (i8*, i8*)*, i8*)

declare void @llvm_hpvm_streamPush(i8*, i8*)

declare i8* @llvm_hpvm_streamPop(i8*)

declare void @llvm_hpvm_streamWait(i8*)

declare i8* @llvm_hpvm_createBindInBuffer(i8*, i64, i32)

declare i8* @llvm_hpvm_createBindOutBuffer(i8*, i64)

declare i8* @llvm_hpvm_createEdgeBuffer(i8*, i64)

declare i8* @llvm_hpvm_createLastInputBuffer(i8*, i64)

declare void @llvm_hpvm_createThread(i8*, i8* (i8*)*, i8*)

declare void @llvm_hpvm_bufferPush(i8*, i64)

declare i64 @llvm_hpvm_bufferPop(i8*)

declare void @llvm_hpvm_cpu_dstack_push(i32, i64, i64, i64, i64, i64, i64)

declare void @llvm_hpvm_cpu_dstack_pop()

declare i64 @llvm_hpvm_cpu_getDimLimit(i32, i32)

declare i64 @llvm_hpvm_cpu_getDimInstance(i32, i32)

declare i8* @llvm_hpvm_initializeTimerSet()

declare void @llvm_hpvm_switchToTimer(i8**, i32)

declare void @llvm_hpvm_printTimerSet(i8**, i8*)

; Function Attrs: nounwind uwtable
define dso_local %struct.out._Z10var_0_nodePvmS_m @_Z10var_0_nodePvmS_m_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned(i8* %t1, i64 %bytes_t1, i8* %t2, i64 %bytes_t2, i64 %idx_x, i64 %idx_y, i64 %idx_z, i64 %dim_x, i64 %dim_y, i64 %dim_z) #4 {
entry:
  %0 = call i8* @wrapper_tensorAdd(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @1, i32 0, i32 0), i8* %t1, i8* %t2)
  %returnStruct = insertvalue %struct.out._Z10var_0_nodePvmS_m undef, i8* %t1, 0
  %returnStruct2 = insertvalue %struct.out._Z10var_0_nodePvmS_m %returnStruct, i64 0, 1
  ret %struct.out._Z10var_0_nodePvmS_m %returnStruct2
}

; Function Attrs: nounwind uwtable
define dso_local %struct.out._Z10var_1_nodePvm @_Z10var_1_nodePvm_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned(i8* %t1, i64 %bytes_t1, i64 %idx_x, i64 %idx_y, i64 %idx_z, i64 %dim_x, i64 %dim_y, i64 %dim_z) #4 {
entry:
  %0 = call i8* @wrapper_tensorRelu(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @2, i32 0, i32 0), i8* %t1)
  %returnStruct = insertvalue %struct.out._Z10var_1_nodePvm undef, i8* %t1, 0
  %returnStruct2 = insertvalue %struct.out._Z10var_1_nodePvm %returnStruct, i64 0, 1
  ret %struct.out._Z10var_1_nodePvm %returnStruct2
}

; Function Attrs: nounwind uwtable
define dso_local %struct.out._Z10var_2_nodePvm @_Z10var_2_nodePvm_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned(i8* %t1, i64 %bytes_t1, i64 %idx_x, i64 %idx_y, i64 %idx_z, i64 %dim_x, i64 %dim_y, i64 %dim_z) #4 {
entry:
  %0 = call i8* @wrapper_tensorTanh(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @3, i32 0, i32 0), i8* %t1)
  %returnStruct = insertvalue %struct.out._Z10var_2_nodePvm undef, i8* %t1, 0
  %returnStruct2 = insertvalue %struct.out._Z10var_2_nodePvm %returnStruct, i64 0, 1
  ret %struct.out._Z10var_2_nodePvm %returnStruct2
}

define %struct.out._Z4rootPvmS_m @_Z4rootPvmS_m_cloned.4(i8* %input1, i64 %input1_bytes, i8* %input2, i64 %input2_bytes) {
entry:
  call void @llvm_hpvm_cpu_dstack_push(i32 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  %_Z10var_0_nodePvmS_m_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned_output = call %struct.out._Z10var_0_nodePvmS_m @_Z10var_0_nodePvmS_m_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned(i8* %input1, i64 %input1_bytes, i8* %input2, i64 %input2_bytes, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  call void @llvm_hpvm_cpu_dstack_pop()
  %0 = extractvalue %struct.out._Z10var_0_nodePvmS_m %_Z10var_0_nodePvmS_m_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned_output, 0
  %1 = extractvalue %struct.out._Z10var_0_nodePvmS_m %_Z10var_0_nodePvmS_m_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned_output, 1
  call void @llvm_hpvm_cpu_dstack_push(i32 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  %_Z10var_1_nodePvm_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned_output = call %struct.out._Z10var_1_nodePvm @_Z10var_1_nodePvm_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned(i8* %0, i64 %1, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  call void @llvm_hpvm_cpu_dstack_pop()
  %2 = extractvalue %struct.out._Z10var_1_nodePvm %_Z10var_1_nodePvm_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned_output, 0
  %3 = extractvalue %struct.out._Z10var_1_nodePvm %_Z10var_1_nodePvm_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned_output, 1
  call void @llvm_hpvm_cpu_dstack_push(i32 0, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  %_Z10var_2_nodePvm_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned_output = call %struct.out._Z10var_2_nodePvm @_Z10var_2_nodePvm_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned(i8* %2, i64 %3, i64 0, i64 0, i64 0, i64 0, i64 0, i64 0)
  call void @llvm_hpvm_cpu_dstack_pop()
  %4 = extractvalue %struct.out._Z10var_2_nodePvm %_Z10var_2_nodePvm_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned_output, 0
  %5 = insertvalue %struct.out._Z4rootPvmS_m undef, i8* %4, 0
  %6 = extractvalue %struct.out._Z10var_2_nodePvm %_Z10var_2_nodePvm_cloned_wrapper_api_cloned_cloned_cloned_cloned_cloned_cloned_output, 1
  %output = insertvalue %struct.out._Z4rootPvmS_m %5, i64 %6, 1
  ret %struct.out._Z4rootPvmS_m %output
}

define i8* @LaunchDataflowGraph(i8* %data.addr) {
entry:
  %input1.addr = bitcast i8* %data.addr to i8**
  %input1 = load i8*, i8** %input1.addr
  %nextArg = getelementptr i8*, i8** %input1.addr, i64 1
  %input1_bytes.addr = bitcast i8** %nextArg to i64*
  %input1_bytes = load i64, i64* %input1_bytes.addr
  %nextArg1 = getelementptr i64, i64* %input1_bytes.addr, i64 1
  %input2.addr = bitcast i64* %nextArg1 to i8**
  %input2 = load i8*, i8** %input2.addr
  %nextArg2 = getelementptr i8*, i8** %input2.addr, i64 1
  %input2_bytes.addr = bitcast i8** %nextArg2 to i64*
  %input2_bytes = load i64, i64* %input2_bytes.addr
  %_Z4rootPvmS_m_cloned.4.output = call %struct.out._Z4rootPvmS_m @_Z4rootPvmS_m_cloned.4(i8* %input1, i64 %input1_bytes, i8* %input2, i64 %input2_bytes)
  %argStructCast.addr = bitcast i8* %data.addr to %_Z4rootPvmS_m_cloned.4.arg.struct.ty*
  %_Z4rootPvmS_m_cloned.4.output.addr = getelementptr %_Z4rootPvmS_m_cloned.4.arg.struct.ty, %_Z4rootPvmS_m_cloned.4.arg.struct.ty* %argStructCast.addr, i32 0, i32 4
  store %struct.out._Z4rootPvmS_m %_Z4rootPvmS_m_cloned.4.output, %struct.out._Z4rootPvmS_m* %_Z4rootPvmS_m_cloned.4.output.addr
  ret i8* null
}

attributes #0 = { uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nofree nounwind }
attributes #4 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { argmemonly nounwind }
attributes #6 = { nofree nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nofree norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #10 = { inlinehint uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #11 = { norecurse nounwind readnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { inlinehint nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #13 = { norecurse uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #14 = { noinline noreturn nounwind }
attributes #15 = { nobuiltin nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #16 = { noreturn "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #17 = { argmemonly nofree nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #18 = { nobuiltin nofree "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #19 = { nounwind readnone speculatable }
attributes #20 = { nounwind }
attributes #21 = { noreturn nounwind }
attributes #22 = { noreturn }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}
!hpvm_hint_promise = !{!2, !3, !4}
!hpvm_hint_gpu = !{}
!hpvm_hint_cpu = !{!5}
!hpvm_hint_cpu_gpu = !{}
!hpvm_hint_cudnn = !{}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 9.0.0 (https://github.com/vzyrianov/hpvm-autograd.git 92936dc9b55dddfc4998228f3ab056ec2a08e385)"}
!2 = !{%struct.out._Z10var_0_nodePvmS_m (i8*, i64, i8*, i64)* undef}
!3 = !{%struct.out._Z10var_1_nodePvm (i8*, i64)* undef}
!4 = !{%struct.out._Z10var_2_nodePvm (i8*, i64)* undef}
!5 = !{%struct.out._Z4rootPvmS_m (i8*, i64, i8*, i64)* undef}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTSNSt12_Vector_baseIfSaIfEE17_Vector_impl_dataE", !8, i64 0, !8, i64 8, !8, i64 16}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!7, !8, i64 8}
!12 = !{!13, !8, i64 56}
!13 = !{!"_ZTS6Tensor", !14, i64 0, !14, i64 4, !14, i64 8, !15, i64 12, !8, i64 16, !8, i64 24, !8, i64 32, !8, i64 40, !8, i64 48, !8, i64 56, !8, i64 64, !16, i64 72, !16, i64 80, !17, i64 88}
!14 = !{!"int", !9, i64 0}
!15 = !{!"_ZTS15data_location_t", !9, i64 0}
!16 = !{!"long", !9, i64 0}
!17 = !{!"_ZTS9Dimension", !14, i64 0, !8, i64 8}
!18 = !{!13, !14, i64 88}
!19 = !{!13, !8, i64 96}
!20 = !{!16, !16, i64 0}
!21 = !{!13, !16, i64 72}
!22 = !{!13, !8, i64 48}
!23 = !{!13, !16, i64 80}
!24 = !{!13, !14, i64 0}
!25 = !{!26, !26, i64 0}
!26 = !{!"float", !9, i64 0}
!27 = !{!14, !14, i64 0}
!28 = !{!29, !29, i64 0}
!29 = !{!"vtable pointer", !10, i64 0}
!30 = !{!31, !16, i64 8}
!31 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !32, i64 0, !16, i64 8, !9, i64 16}
!32 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !8, i64 0}
!33 = !{!34, !26, i64 0}
!34 = !{!"_ZTS9ClassProb", !26, i64 0, !14, i64 4}
!35 = !{!34, !14, i64 4}
!36 = !{!9, !9, i64 0}
!37 = !{!38, !8, i64 8}
!38 = !{!"_ZTSNSt12_Vector_baseI9ClassProbSaIS0_EE17_Vector_impl_dataE", !8, i64 0, !8, i64 8, !8, i64 16}
!39 = !{!38, !8, i64 16}
!40 = !{!38, !8, i64 0}
!41 = !{!7, !8, i64 16}
!42 = !{!43, !8, i64 0}
!43 = !{!"_ZTS6RootIn", !8, i64 0, !16, i64 8, !8, i64 16, !16, i64 24, !44, i64 32}
!44 = !{!"_ZTS5ret_t", !8, i64 0, !16, i64 8}
!45 = !{!43, !16, i64 8}
!46 = !{!43, !8, i64 16}
!47 = !{!43, !16, i64 24}
!48 = !{!43, !8, i64 32}
!49 = !{!50, !51, i64 24}
!50 = !{!"_ZTSSt8ios_base", !16, i64 8, !16, i64 16, !51, i64 24, !52, i64 28, !52, i64 32, !8, i64 40, !53, i64 48, !9, i64 64, !14, i64 192, !8, i64 200, !54, i64 208}
!51 = !{!"_ZTSSt13_Ios_Fmtflags", !9, i64 0}
!52 = !{!"_ZTSSt12_Ios_Iostate", !9, i64 0}
!53 = !{!"_ZTSNSt8ios_base6_WordsE", !8, i64 0, !16, i64 8}
!54 = !{!"_ZTSSt6locale", !8, i64 0}
!55 = !{!51, !51, i64 0}
!56 = !{!31, !8, i64 0}
!57 = !{!32, !8, i64 0}
!58 = !{!8, !8, i64 0}
!59 = !{!60, !8, i64 216}
!60 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !8, i64 216, !9, i64 224, !61, i64 225, !8, i64 232, !8, i64 240, !8, i64 248, !8, i64 256}
!61 = !{!"bool", !9, i64 0}
!62 = !{!60, !9, i64 224}
!63 = !{!60, !61, i64 225}
!64 = !{!65, !66, i64 64}
!65 = !{!"_ZTSNSt7__cxx1115basic_stringbufIcSt11char_traitsIcESaIcEEE", !66, i64 64, !31, i64 72}
!66 = !{!"_ZTSSt13_Ios_Openmode", !9, i64 0}
!67 = !{!68, !8, i64 40}
!68 = !{!"_ZTSSt15basic_streambufIcSt11char_traitsIcEE", !8, i64 8, !8, i64 16, !8, i64 24, !8, i64 32, !8, i64 40, !8, i64 48, !54, i64 56}
!69 = !{!68, !8, i64 24}
!70 = !{!68, !8, i64 32}
!71 = !{!72, !8, i64 0}
!72 = !{!"_ZTSN9__gnu_cxx17__normal_iteratorIPKcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE", !8, i64 0}
!73 = !{!74, !8, i64 0}
!74 = !{!"_ZTSN9__gnu_cxx17__normal_iteratorIPcNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEEE", !8, i64 0}
!75 = !{!76, !8, i64 0}
!76 = !{!"_ZTSN9__gnu_cxx17__normal_iteratorIP9ClassProbSt6vectorIS1_SaIS1_EEEE", !8, i64 0}
!77 = !{i64 0, i64 65}
!78 = !{!79, !8, i64 0}
!79 = !{!"_ZTSN9__gnu_cxx5__ops15_Iter_comp_iterIPFb9ClassProbS2_EEE", !8, i64 0}
!80 = !{!81, !8, i64 0}
!81 = !{!"_ZTSN9__gnu_cxx5__ops14_Iter_comp_valIPFb9ClassProbS2_EEE", !8, i64 0}
!82 = !{!83, !8, i64 0}
!83 = !{!"_ZTSN9__gnu_cxx5__ops14_Val_comp_iterIPFb9ClassProbS2_EEE", !8, i64 0}
!84 = !{!85, !8, i64 0}
!85 = !{!"_ZTSN9__gnu_cxx17__normal_iteratorIPfSt6vectorIfSaIfEEEE", !8, i64 0}
!86 = !{!50, !52, i64 32}
!87 = !{!60, !8, i64 240}
!88 = !{!89, !9, i64 56}
!89 = !{!"_ZTSSt5ctypeIcE", !8, i64 16, !61, i64 24, !8, i64 32, !8, i64 40, !8, i64 48, !9, i64 56, !9, i64 57, !9, i64 313, !9, i64 569}
