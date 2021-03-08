#include "Fit/Fitter.h"
#include "Fit/BinData.h"
#include "Fit/Chi2FCN.h"
#include "TH1.h"
#include "TList.h"
#include "Math/WrappedMultiTF1.h"
#include "HFitInterface.h"
#include "TCanvas.h"
#include "TStyle.h"
#include <string>

using std::string;

// definition of shared parameter
// background function
int iparB[5] = { 0,   // bkg normalisation B
                 2,   // fraction of exp-gaus common
                 3,   // exp common parameter
                 4,   // gaus mean bkg
                 5    // gaus sigma bkg
};
// signal + background function
int iparSB[8] = {1,   // bkg normalisation SB
                 2,   // fraction of exp-gaus common
                 3,   // exp common parameter
                 4,   // gaus mean bkg
                 5,    // gaus sigma bkg
                 6,   // gaus ampl signal
                 7,   // gaus mean signal
                 8   // gaus sigma signal
};

// Create the GlobalCHi2 structure
struct GlobalChi2 {
   GlobalChi2(  ROOT::Math::IMultiGenFunction & f1,
                ROOT::Math::IMultiGenFunction & f2) :
      fChi2_1(&f1), fChi2_2(&f2) {}
   // parameter vector is first background (in common 1 and 2)
   // and then is signal (only in 2)
   double operator() (const double *par) const {
      double p1[5];
      for (int i = 0; i < 5; ++i) p1[i] = par[iparB[i] ];
      double p2[8];
      for (int i = 0; i < 8; ++i) p2[i] = par[iparSB[i] ];
      return (*fChi2_1)(p1) + (*fChi2_2)(p2);
   }

   const  ROOT::Math::IMultiGenFunction * fChi2_1;
   const  ROOT::Math::IMultiGenFunction * fChi2_2;
};

double comb_fit_gaus(TH1D * hB, TH1D * hSB, string name, double xmin, double xmax, double mcsigma=-1) {
 
   TF1 * fB = new TF1("fB","[0]*([1]*exp([2]*x) + (1-[1])*exp(-0.5*((x-[3])/[4])**2)/(sqrt(2*pi)*[4]))", xmin, xmax);
   fB->SetParameters(1,-0.05);
 
 
   // perform now global fit
 
   TF1 * fSB = new TF1("fSB","[0]*([1]*exp([2]*x) + (1-[1])*exp(-0.5*((x-[3])/[4])**2)/(sqrt(2*pi)*[4]))+ gausn(5)", xmin, xmax); 
   fSB->SetNpx(300);
 
   ROOT::Math::WrappedMultiTF1 wfB(*fB,1);
   ROOT::Math::WrappedMultiTF1 wfSB(*fSB,1);
 
   ROOT::Fit::DataOptions opt;
   ROOT::Fit::DataRange rangeB;
   // set the data range
   rangeB.SetRange(xmin, xmax);
   ROOT::Fit::BinData dataB(opt,rangeB);
   ROOT::Fit::FillData(dataB, hB);
 
   ROOT::Fit::DataRange rangeSB;
   rangeSB.SetRange(xmin, xmax);
   ROOT::Fit::BinData dataSB(opt,rangeSB);
   ROOT::Fit::FillData(dataSB, hSB);
 
   ROOT::Fit::Chi2Function chi2_B(dataB, wfB);
   ROOT::Fit::Chi2Function chi2_SB(dataSB, wfSB);
 
   GlobalChi2 globalChi2(chi2_B, chi2_SB);
 
   ROOT::Fit::Fitter fitter;
 
   const int Npar = 9;

   float sigma_start = mcsigma > 0 ? mcsigma : 0.002; 
   double par0[Npar] = { 5, 5, 0.1, 0.1, 2.994, 0.004 , 30, 2.991, sigma_start};
 
   // create before the parameter settings in order to fix or set range on them
   fitter.Config().SetParamsSettings(9, par0);

   // fix 5-th parameter
   // set limits on the third and 4-th parameter
   fitter.Config().ParSettings(0).SetLimits(0, 1000);
   fitter.Config().ParSettings(1).SetLimits(0, 1000);
   fitter.Config().ParSettings(2).SetLimits(-10, 10);
   fitter.Config().ParSettings(3).SetLimits(0, 1);
   fitter.Config().ParSettings(4).SetLimits(2.990, 2.996);
   fitter.Config().ParSettings(5).SetLimits(0.002, 0.006);
   fitter.Config().ParSettings(6).SetLimits(0,1000);
   fitter.Config().ParSettings(7).SetLimits(2.989, 3.0);

   mcsigma!=-1 ? fitter.Config().ParSettings(8).Fix() : fitter.Config().ParSettings(8).SetLimits(0.001,0.003);
 
 
   fitter.Config().MinimizerOptions().SetPrintLevel(0);
   fitter.Config().SetMinimizer("Minuit2","Migrad");
 
   // fit FCN function directly
   // (specify optionally data size and flag to indicate that is a chi2 fit)
   fitter.FitFCN(9,globalChi2,0,dataB.Size()+dataSB.Size(),true);
   ROOT::Fit::FitResult result = fitter.Result();
   result.Print(std::cout);
 
   TCanvas * c1 = new TCanvas();
   c1->SetName(name.data());
   c1->Divide(1,2);
   c1->cd(1);
   gStyle->SetOptFit(1111);
 
   fB->SetFitResult( result, iparB);
   fB->SetRange(rangeB().first, rangeB().second);
   fB->SetLineColor(kBlue);
   hB->GetListOfFunctions()->Add(fB);
   hB->Draw("PE");
 
   c1->cd(2);
   fSB->SetFitResult( result, iparSB);
   fSB->SetRange(rangeSB().first, rangeSB().second);
   fSB->SetLineColor(kRed);
   hSB->GetListOfFunctions()->Add(fSB);
   hSB->Draw("PE");
   fB->SetLineColor(kBlue);
   fB->SetLineStyle(kDashed);
   fB->Draw("SAME");
   

   c1->Write();
   return result.Parameter(6)/hSB->GetBinWidth(1);
}