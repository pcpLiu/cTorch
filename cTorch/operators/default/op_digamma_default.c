#include <tgmath.h>

#include "cTorch/operators/default/op_list.h"
#include "cTorch/operators/default/util.h"

#ifndef M_PIl
/** The constant Pi in high precision */
#define M_PIl 3.1415926535897932384626433832795029L
#endif
#ifndef M_GAMMAl
/** Euler's constant in high precision */
#define M_GAMMAl 0.5772156649015328606065120900824024L
#endif
#ifndef M_LN2l
/** the natural logarithm of 2 in high precision */
#define M_LN2l 0.6931471805599453094172321214581766L
#endif

/** The digamma function in long double precision.
 * @param x the real value of the argument
 * @return the value of the digamma (psi) function at that point
 * @author Richard J. Mathar
 * @since 2005-11-24
 *
 * @ref https://www2.mpia-hd.mpg.de/~mathar/progs/digamma.c
 */
long double digammal(long double x) {
  /* force into the interval 1..3 */
  if (x < 0.0L)
    return digammal(1.0L - x) +
           M_PIl / tanl(M_PIl * (1.0L - x)); /* reflection formula */
  else if (x < 1.0L)
    return digammal(1.0L + x) - 1.0L / x;
  else if (x == 1.0L)
    return -M_GAMMAl;
  else if (x == 2.0L)
    return 1.0L - M_GAMMAl;
  else if (x == 3.0L)
    return 1.5L - M_GAMMAl;
  else if (x > 3.0L)
    /* duplication formula */
    return 0.5L * (digammal(x / 2.0L) + digammal((x + 1.0L) / 2.0L)) + M_LN2l;
  else {
    /* Just for your information, the following lines contain
     * the Maple source code to re-generate the table that is
     * eventually becoming the Kncoe[] array below
     * interface(prettyprint=0) :
     * Digits := 63 :
     * r := 0 :
     *
     * for l from 1 to 60 do
     * 	d := binomial(-1/2,l) :
     * 	r := r+d*(-1)^l*(Zeta(2*l+1) -1) ;
     * 	evalf(r) ;
     * 	print(%,evalf(1+Psi(1)-r)) ;
     *o d :
     *
     * for N from 1 to 28 do
     * 	r := 0 :
     * 	n := N-1 :
     *
     *	for l from iquo(n+3,2) to 70 do
     *		d := 0 :
     *		for s from 0 to n+1 do
     *		 d := d+(-1)^s*binomial(n+1,s)*binomial((s-1)/2,l) :
     *		od :
     *		if 2*l-n > 1 then
     *		r := r+d*(-1)^l*(Zeta(2*l-n) -1) :
     *		fi :
     *	od :
     *	print(evalf((-1)^n*2*r)) ;
     *od :
     *quit :
     */
    static long double Kncoe[] = {.30459198558715155634315638246624251L,
                                  .72037977439182833573548891941219706L,
                                  -.12454959243861367729528855995001087L,
                                  .27769457331927827002810119567456810e-1L,
                                  -.67762371439822456447373550186163070e-2L,
                                  .17238755142247705209823876688592170e-2L,
                                  -.44817699064252933515310345718960928e-3L,
                                  .11793660000155572716272710617753373e-3L,
                                  -.31253894280980134452125172274246963e-4L,
                                  .83173997012173283398932708991137488e-5L,
                                  -.22191427643780045431149221890172210e-5L,
                                  .59302266729329346291029599913617915e-6L,
                                  -.15863051191470655433559920279603632e-6L,
                                  .42459203983193603241777510648681429e-7L,
                                  -.11369129616951114238848106591780146e-7L,
                                  .304502217295931698401459168423403510e-8L,
                                  -.81568455080753152802915013641723686e-9L,
                                  .21852324749975455125936715817306383e-9L,
                                  -.58546491441689515680751900276454407e-10L,
                                  .15686348450871204869813586459513648e-10L,
                                  -.42029496273143231373796179302482033e-11L,
                                  .11261435719264907097227520956710754e-11L,
                                  -.30174353636860279765375177200637590e-12L,
                                  .80850955256389526647406571868193768e-13L,
                                  -.21663779809421233144009565199997351e-13L,
                                  .58047634271339391495076374966835526e-14L,
                                  -.15553767189204733561108869588173845e-14L,
                                  .41676108598040807753707828039353330e-15L,
                                  -.11167065064221317094734023242188463e-15L};

    register long double Tn_1 = 1.0L;   /* T_{n-1}(x), started at n=1 */
    register long double Tn = x - 2.0L; /* T_{n}(x) , started at n=1 */
    register long double resul = Kncoe[0] + Kncoe[1] * Tn;

    x -= 2.0L;

    for (int n = 2; n < sizeof(Kncoe) / sizeof(long double); n++) {
      const long double Tn1 =
          2.0L * x * Tn -
          Tn_1; /* Chebyshev recursion, Eq. 22.7.4 Abramowitz-Stegun */
      resul += Kncoe[n] * Tn1;
      Tn_1 = Tn;
      Tn = Tn1;
    }
    return resul;
  }
}

#define _cth_digamma_kernel(input_ptr, output_ptr, N, data_type)               \
  do {                                                                         \
    data_type *input_t = (data_type *)input_ptr;                               \
    data_type *output_t = (data_type *)output_ptr;                             \
    for (int i = 0; i < N; i++) {                                              \
      output_t[i] = (data_type)digammal(input_t[i]);                           \
    }                                                                          \
  } while (0)

/**
 * Computes the logarithmic derivative of the gamma function on input.
 *
 * Inputs & outputs:
 *    - # of input: 1
 *    - # of output: 1
 */
void op_digamma_cpu(CTorchOperator *op) {
  FORCE_OP_INPUT_OUTPUT_TENSOR_NUM(op, 1, 1);
  OP_FAIL_ON_DTYPE(op, CTH_TENSOR_DATA_TYPE_BOOL);

  CTorchTensor *in = array_at(CTorchTensor)(op->in_bound_tensors, 0);
  CTorchTensor *out = array_at(CTorchTensor)(op->out_bound_tensors, 0);
  tensor_size_t N = in->meta_info->n_elements;
  _cpu_1d_map_elewise_unary_generic(
      in->values,
      out->values,
      in->meta_info->data_type,
      N,
      _cth_digamma_kernel);
}
