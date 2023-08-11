*______________________________________________________________________________;
*                                                                              ;
* ECHO.READIN.SAS                                                              ;
* NOTE - Wall motion variables added 8/13/2014 - EAL                           ;
* NOTE - Added additional labels to variables 8/22/2014 - EAL                  ;
* NOTE - Code added to allow taking the most recent, non-missing measure       ;
*        for preop echos -MS                                                   ;
* NOTE - Labels added for a few preop measures 04/15/2015 -MS                  ;
* NOTE - Cleaned up code that was put in for the testing of the new code       ;
*        4/15/2015 - EAL                                                       ;
* NOTE - Wall motion variables commented out for the old warehouse since these ;
*        are sparsely populated and we rarely receive them in our pulls        ;
*        11/29/2016 - MS                                                       ;
* NOTE - Adjusting for warehouse echo pull 11/04/19 -MS                        ;
* NOTE - Code adapted to new echo warehouse 05/19/2020 -MS
* NOTE - Added units to echo measures 11/4/2020
* NOTE - Added units to preop echo measures   11/30/2020 -MS            ;
*______________________________________________________________________________;
*                                                                              ;
*______________________________________________________________________________;
*                                                                              ;
* Set macro variables to define directory path and study description           ;

 %let STUDY      =/studies/cardiac/xxxx;
 %let STUDYDESC  =Study Description;


*______________________________________________________________________________;
*                                                                              ;
* NOTE: Use only 8-character variable names and 40-character labels for        ;
*       compatability with hazard program                                      ;
*______________________________________________________________________________;
*                                                                              ;
 options pagesize=107 linesize=132 nofmterr nocenter;
 libname library "&STUDY/datasets";

 title1 "&STUDYDESC";
 title3 "Echo Data Set";



*******************************************;
*   CODE FOR NEW DATA WAREHOUSE PULL       ;
*******************************************;
*keep echo variables needed for your study;
data echo1   ;
 set library.stNUM_echo;

 keep
  ccfid
  dtn_inst
  dtn_echo
  echo_plc
  echotype
  modality
  ef_echo
  lvidd
  lvisd
  ivswt
  ladia
  pwt
  lvedvol
  lvesvol
  lv_mass
  aoszascm
  aosin_dm
  av_regn
  av_stenn
  avpkgrad
  avmngrad
  av_area
  lvotpkgr
  lvotmngr
  lvotpkga
  lvotmnga
  tv_regn
  tv_rvel
  mv_regn
  mv_stenn
  mvpkgrad
  mvmngrad
  las_area
  lad_area
  la_throm
  rvsp
  apicsept
  apic_inf
  apic_lat
  apic_ant
  bas_ants
  bas_sept
  bas_infe
  bas_post
  bas_late
  bas_ante
  midantse
  mid_sept
  mid_infe
  mid_post
  mid_late
  mid_ante
  di_fun
  di_funrc
  di_funwm
  di_funm
  di_funfp
  di_funcd
  lvdpex
  sys_funn
  sys_fcnp
  ras_area
  rad_area
  rvs_area
  rvd_area
  rvesvol
  rvedvol
  av_vti
   ;

run;


*make sure you have all of the identifiers, dates and datetime variables you need;
data echo2;
  set echo1;

  dt_surg = datepart(dtn_inst);
  format dt_surg mmddyy10.;

  echo_dt=datepart(dtn_echo);
  dtm_echo = dtn_echo;

  format echo_dt mmddyy10.;
  format dtm_echo datetime.;


run;


*process echo data;
data echo3;
  set echo2 (rename = (sys_funn=sys_func));


%macro skip;
*do not go through this conversion since we want to maintain text responses;

  *** Convert text valve regurg and stenosis values to CVIR numeric rank order;
  array convert(5) av_regur mv_regur tv_regur av_sten mv_sten;
  array output(5) av_regn mv_regn tv_regn av_stenn mv_stenn;

    i=1;
      do i=1 to 5;
        if convert(i) = "1+" then output(i) = 1;
        if convert(i) = "1+2+" then output(i) = 2;
        if convert(i) = "2+" then output(i) = 2;
        if convert(i) = "2+3+" then output(i) = 3;
        if convert(i) = "3+" then output(i) = 3;
        if convert(i) = "3+4+" then output(i) = 4;
        if convert(i) = "4+" then output(i) = 4;
        if convert(i) = "NONE" then output(i) = 0;
        if convert(i) = "none" then output(i) = 0;
        if convert(i) = "T1+" then output(i) = 1;
        if convert(i) = "TRIV" then output(i) = 0;
        if convert(i) = "TRIVIAL" then output(i) = 0;
        if convert(i) = "TRIVIAL;" then output(i) = 0;
        if convert(i) = "CRIT" then output(i) = 5;
        if convert(i) = "MILD" then output(i) = 1;
        if convert(i) = "MILD; MEASURED MITRAL VALVE ORIFACE" then output(i) = 1;
        if convert(i) = "MEASURED MITRAL VALVE ORIFACE" then output(i) = .;
        if convert(i) = "MILDLY;" then output(i) = 1;
        if convert(i) = "PULSED; CONTINUOUS; MILD 1+;" then output(i) = 1;
        if convert(i) = "PULSED; MILD 1+;" then output(i) = 1;
        if convert(i) = "CONTINUOUS;" then output(i) = .;
        if convert(i) = "CONTINUOUS WAVE;" then output(i) = .;
        if convert(i) = "CONTINUOUS WAVE; MILD 1+;" then output(i) = 1;
        if convert(i) = "CONTINUOUS; MILD 1+;" then output(i) = 1;
        if convert(i) = "CONTINUOUS; MODERATE 2+;" then output(i) = 2;
        if convert(i) = "CONTINUOUS WAVE; MODERATE 2+;" then output(i) = 2;
        if convert(i) = "CONTINUOUS; MODERATELY SEVERE 3+;" then output(i) = 3;
        if convert(i) = "CONTINUOUS WAVE; MODERATELY SEVERE 3+;" then output(i) = 3;
        if convert(i) = "CONTINUOUS; SEVERE 4+;" then output(i) = 4;
        if convert(i) = "CONTINUOUS WAVE; SEVERE 4+;" then output(i) = 4;
        if convert(i) = "MILD 1+;" then output(i) = 1;
        if convert(i) = "MILD 1+; PEAK VELOCITY;" then output(i) = 1;
        if convert(i) = "MILD 1+; PRESSURE HALF TIME (T1/2);" then output(i) = 1;
        if convert(i) = "PRESSURE HALF TIME (T1/2);" then output(i) = .;
        if convert(i) = "MILD 1+; PARAVALVULAR;" then output(i) = 1;
        if convert(i) = "PARAVALVULAR;" then output(i) = .;
        if convert(i) = "PULSED; CONTINUOUS WAVE;" then output(i) = .;
        if convert(i) = "PULSED; CONTINUOUS;" then output(i) = .;
        if convert(i) = "PULSED; CONTINUOUS WAVE; MILD 1+;" then output(i) = 1;
        if convert(i) = "PEAK VELOCITY;" then output(i) = .;
        if convert(i) = "PEAK VELOCITY; TRIVIAL;" then output(i) = 0;
        if convert(i) = "MILD 1+; MODERATE 2+;" then output(i) = 2;
        if convert(i) = "MILD 1+; MODERATE 2+; PEAK VELOCITY;" then output(i) = 2;
        if convert(i) = "MOD" then output(i) = 3;
        if convert(i) = "MODERATE" then output(i) = 3;
        if convert(i) = "MODERATE; MEASURED MITRAL VALVE ORIFACE" then output(i) = 3;
        if convert(i) = "MODERATELY;" then output(i) = 3;
        if convert(i) = "MODERATE 2+;" then output(i) = 2;
        if convert(i) = "MODERATE 2+; PARAVALVULAR;" then output(i) = 2;
        if convert(i) = "MODERATE 2+; PEAK VELOCITY;" then output(i) = 2;
        if convert(i) = "PULSED; MODERATE 2+;" then output(i) = 2;
        if convert(i) = "PULSED; CONTINUOUS; MODERATE 2+;" then output(i) = 2;
        if convert(i) = "PULSED; CONTINUOUS WAVE; MODERATE 2+;" then output(i) = 2;
        if convert(i) = "MODERATE 2+; PRESSURE HALF TIME (T1/2);" then output(i) = 2;
        if convert(i) = "MODERATE 2+; MODERATELY SEVERE 3+;" then output(i) = 3;
        if convert(i) = "MODERATE 2+; MODERATELY SEVERE 3+; PEAK VELOCITY;" then output(i) = 3;
        if convert(i) = "MODERATELY SEVERE 3+;" then output(i) = 3;
        if convert(i) = "PULSED; CONTINUOUS WAVE; MODERATELY SEVERE 3+;" then output(i) = 3;
        if convert(i) = "PULSED; CONTINUOUS; MODERATELY SEVERE 3+;" then output(i) = 3;
        if convert(i) = "PULSED; MODERATELY SEVERE 3+;" then output(i) = 3;
        if convert(i) = "MODERATELY SEVERE 3+; PEAK VELOCITY;" then output(i) = 3;
        if convert(i) = "MODERATELY SEVERE 3+; PRESSURE HALF TIME (T1/2);" then output(i) = 3;
        if convert(i) = "MODERATELY SEVERE 3+; PARAVALVULAR;" then output(i) = 3;
        if convert(i) = "MODERATELY SEVERE 3+; SEVERE 4+;" then output(i) = 4;
        if convert(i) = "MODERATELY SEVERE 3+; SEVERE 4+; PEAK VELOCITY;" then output(i) = 4;
        if convert(i) = "PULSED;" then output(i) = .;
        if convert(i) = "PULSED; SEVERE 4+;" then output(i) = 4;
        if convert(i) = "PULSED; CONTINUOUS WAVE; SEVERE 4+;" then output(i) = 4;
        if convert(i) = "PULSED; CONTINUOUS; SEVERE 4+;" then output(i) = 4;
        if convert(i) = "SEVERE 4+;" then output(i) = 4;
        if convert(i) = "SEVERE 4+; PRESSURE HALF TIME (T1/2);" then output(i) = 4;
        if convert(i) = "SEVERE 4+; PEAK VELOCITY;" then output(i) = 4;
        if convert(i) = "SEVERE 4+; PARAVALVULAR;" then output(i) = 4;
        if convert(i) = "MSEV" then output(i) = 4;
        if convert(i) = "N" then output(i) = 0;
        if convert(i) = "SEV" then output(i) = 5;
        if convert(i) = "SEVERE" then output(i) = 5;
        if convert(i) = "SEVERE; MEASURED MITRAL VALVE ORIFACE" then output(i) = 5;
        if convert(i) = "SEVERELY;" then output(i) = 5;
        if convert(i) = "Y" then output(i) = 5;
        if convert(i) = "ABN" then output(i) = 2;
        if convert(i) = "NORM" then output(i) = 1;
        if convert(i) = "S1" then output(i) = 1;
        if convert(i) = "S2" then output(i) = 2;
        if convert(i) = "S3" then output(i) = 3;
        if convert(i) = "S4" then output(i) = 4;
        if convert(i) = "HYPE" then output(i) = 1;
        if convert(i) = "LNORM" then output(i) = 1;
      end;

    if av_sten="critical" then av_stenn=5;
    if av_sten="moderate_severe" then av_stenn=4;

    drop av_regur mv_regur tv_regur av_sten mv_sten sys_func i;
%mend skip;


    if sys_func in ("HYPERDYNAMIC" "LOW NORMAL" "NORMAL") then sys_funn=1;
      else if sys_func="MILD DECREASE" then sys_funn=2;
      else if sys_func="MODERATE DECREASE" then sys_funn=3;
      else if sys_func="MODERATELY SEVERE DECREASE" then sys_funn=4;
      else if sys_func="SEVERE DECREASE" then sys_funn=5;



  *** Delete echos with no data;
  if ladia=. and lvidd=. and lvisd=. and pwt=. and ivswt=. and ef_echo=.
    and avpkgrad=. and avmngrad=. and av_area=. and
    mvpkgrad=. and mvmngrad=. and tv_rvel=. and las_area=. and lad_area=.
    and av_regn='' and mv_regn='' and tv_regn='' and av_stenn='' and mv_stenn='' and sys_funn=.
  then delete;



  *** Adjust measurement units of old echo left atrial diameters;
  if ladia>=1000 then ladia=ladia/1000;

  label
    ccfid   ='CCF ID'
    dt_surg ='Date: Surgery'
    echo_dt ='Date: Echo'
    dtm_echo='Date and Time of Echo'
    echo_plc='Echo location'
    echotype='Echo type'
    /*echotime='Echo timing'*/
    ladia   ='Left atrial diameter (cm)'
    lvidd   ='Left ventricular inner diameter in diastole (cm)'
    lvisd   ='Left ventricular inner diameter in systole (cm)'
    pwt     ='Posterior wall thickness (cm)'
    ivswt   ='Intraventricular septal thickness (cm)'
    ef_echo ='Ejection fraction (%)'
    av_regn ='AV regurgitation degree'
    av_stenn='AV stenosis degree (mmHg)'
    avpkgrad='AV peak gradient (mmHg)'
    avmngrad='AV mean gradient (mmHg)'
    av_area ='AV area (cm^2)'
    mv_regn ='MV regurgitation degree'
    mv_stenn='MV stenosis degree'
    mvpkgrad='MV peak gradient (mmHg)'
    mvmngrad='MV mean gradient (mmHg)'
    tv_regn ='TV regurgitation degree'
    tv_rvel ='TV regurgitation velocity'
    las_area='Left atrial systolic area (cm^2)'
    lad_area='Left atrial diastolic area (cm^2)'
    lvedvol ='Left ventricular end-diastolic volume (mL)'
    lvesvol ='Left ventricular end-systolic volume(mL)'
    rvsp    ='Right ventricular systolic pressure'
    sys_funn='Left ventricular systolic function-1:NORM,2:MILD,3:MOD,4:MSEV,5:SEV'
    apicsept='Wall Motion: Apical Septum'
    apic_inf='Wall Motion: Apical Inferior'
    apic_lat='Wall Motion: Apical Lateral'
    apic_ant='Wall Motion: Apical Anterior'
    bas_ants='Wall Motion: Basal Anterior Septum'
    bas_sept='Wall Motion: Basal Septum'
    bas_infe='Wall Motion: Basal Inferior'
    bas_post='Wall Motion: Basal Posterior'
    bas_late='Wall Motion: Basal Lateral'
    bas_ante='Wall Motion: Basal Anterior'
    midantse='Wall Motion: Mid Anterior Septum'
    mid_sept='Wall Motion: Mid Septum'
    mid_infe='Wall Motion: Mid Inferior'
    mid_post='Wall Motion: Mid Posterior'
    mid_late='Wall Motion: Mid Lateral'
    mid_ante='Wall Motion: Mid Anterior'
    lv_mass ='LV Mass'
    lvotpkgr='LV Outflow Track Peak Gradient'
    lvotmngr='LV Outflow Track Mean Gradient'
    lvotpkga='LVOT Peak Gradient after Amyl Nitrite'
    lvotmnga='LVOT Mean Gradient after Amyl Nitrite'
    la_throm='Left Atrial Thrombosis'
    di_fun  ='Diastolic Function'
    di_funrc='Diastolic Function Res Con'
    di_funwm='Diastolic Function Filling Pattern w/ Maneuver'
    di_funm ='Diastolic Function Maneuver Type'
    di_funfp='Diastolic Function Filling Pressures'
    di_funcd='Diastolic Func Color Dopper m-Mode Flow Propagation'
    lvdpex  ='LV Dilation Post Exercise'
    sys_fcnp='Systolic Function Pattern'
    ras_area='Right Atrium Systolic Area'
    rad_area='Right Atrium Diastolic Area'
    rvs_area='Right Ventricle Systolic Area'
    rvd_area='Right Ventricle Diastolic Area'
    rvesvol ='Right Ventricle End Systolic Area'
    rvedvol ='Right Ventricle End Diastolic Area'
  ;
run;



*------------------------------------------------------------------------------;
* Create postop multiple echos dataset                                         ;
*------------------------------------------------------------------------------;
data postopechos1;
  set echo3;

  poflag=0;
  /* Per Dr. Blackstone, the echo types that are used for analysis should be limited to plain TTE only.
     In the past, TTE, SE, ARBUT and DOB types were used. (9/7/2011)

     1/9/2014
     Per Dr. Blackstone, the following 3 echo types are very specialized and we are not interested in them
     ECHOC   - Echo with contrast
     ECHOAV  - When a pt comes in with a BiV pacer, they don an AV Optimization echo
     PROECHO - Echo done during a procedure
  */

  if dtm_echo > dtn_inst and
    ((echotype in ('TTE', 'ECHO', 'XECHO')) )
  then poflag=1;

  if poflag=0 then delete; ***Drop the echos >= date and time of incision;


  rename
    aoszascm  =midaa_dm
  ;

  label
    aoszascm = 'Mid ascending aorta diameter'
  ;


  *** Echocardiographic derivatives;
  if lvidd  le lvisd then do;
    lvidd=.; lvisd=. ;
  end;

  *** LV Fractional Shortening;
  fs=(lvidd-lvisd)/lvidd;

  *** LV Relative Wall Thickness - this is a measure of wall stress;
  rwt=2*pwt/lvidd;

  *******************************************************************************;
  * LV Mass                                                                      ;
  * Deveraux RB, Reichek N. Echocardiographic determination of left ventricular  ;
  * mass in man. Anatomic validation of the method. Circulation 1977,55:613-18   ;
  *                                                                              ;
  * Corrected for ASE method and overestimation                                  ;
  * Deveraux RB, Alonso DR, Lutas EM, Gottlieb GJ, Campo E, Sachs I, Reichek N.  ;
  * Echocardiographic assessment of left ventricular hypertrophy: Comparison to  ;
  * necropsy findings.  Am J Cardiol 1986,57:450-8                               ;
  *                                                                              ;
  * Indexing is controversial: straight BSA or BSA to a power (1.5)              ;
  * deSimone et al. J Am Coll Cardiol 1995,25:1056-62                            ;
  * Gutgesell HP. Growth of the human heart relative to body surface area.  Am J ;
  * Cardiol 1990,65:662-8                                                        ;
  *                                                                              ;
  * Units are: wall thickness in mm (millimeters), mass in g (grams)             ;
  *******************************************************************************;

  lvmass=0.80*(1.04*(((lvidd + pwt + ivswt)**3) - lvidd**3)) + 0.6;
  *plvmassi=plvmass/bsa;
 *ms_lvmas=0; *if lvmassi=. then ms_lvmas=1;

  *******************************************************************************;
  * LV Volumes                                                                   ;
  * Teichholz LE, Kreulen T, Herman MI, Gorlin R.  Problems in                   ;
  * exhocardiocardiographic-angiographic correlations in the presence or absence ;
  * of asynergy. Am J Cardiol 1976,37:7-11                                       ;
  *                                                                              ;
  * Volumes are expressed in mL (milliliters) from dimensions in mm (millimeters);
  *******************************************************************************;

  lvedv=(7.0/(2.4 + lvidd))*(lvidd**3);
  lvesv=(7.0/(2.4 + lvisd))*(lvisd**3);
  lvef_ev=100*(lvedv-lvesv)/lvedv;
  *lvedvi=lvedv/bsa; *lvesvi=lvesv/bsa;

  *** LA Volume;
  ***No formula, but assume that it will be a scaled cubed function;
  pi=2*arcos(0);
  lavol=(4*pi/3)*((ladia/2)**3);
  *lavoli=lavol/bsa;
  la_ge6=(ladia ge 6);
  if ladia=. then la_ge6=.;
  drop pi;

  *******************************************************************************;
  * RV systolic pressure                                                         ;
  * Yock PG, Popp RL. Noninvasive estimation of right ventricular systolic       ;
  * pressure by Doppler ultrasound in patients with tricuspid regurgitation.     ;
  * Circulation 1984,70:657-62.                                                  ;
  *                                                                              ;
  * From Bernoulli relation, RV pressure = RA pressure + 4*((TV velocity)^2),    ;
  * where TV velocity is peak tricuspid velocity in m/s (meters/second) and      ;
  * RV and RA pressures are measured in mmHg (millimetes of mercury)             ;
  *                                                                              ;
  * RA pressure may be estimated from jugular pressure or held constant. The     ;
  * latter is undesirable because PA pressure and RA pressure are positively     ;
  * correlated. Therefore:                                                       ;
  *                                                                              ;
  * Berger (JACC 1985,6:359-65) suggests a regression equation without RA        ;
  * pressure that multiplies this quantity by 1.23. Berger M, Haimowitz A,       ;
  * Van Tosh A, Berdoff RL, Goldberg E. Quantitative assessment of pulmonary     ;
  * hypertension in patients with tricuspid regurgitation using continuous wave  ;
  * Doppler ultrasound. J Am Coll Cardiol 1985,6:359-65.                         ;
  *                                                                              ;
  * By regression analysis, Wu used: RVSPNEW = 10 + 4*(TVPKVEL**2)               ;
  *******************************************************************************;

  *** RV systolic pressure extremes have few numbers;
  rvspnew=10 +4*((tv_rvel/100)**2);

  label
    fs      ='Calculated Fractional shortening'
    rwt     ='Calculated LV relative wall thickness'
    lvmass  ='Calculated LV mass (ASE-cube mass, corrected) (g)'
    lvedv   ='Calculated LV end diatolic volume (mL)'
    lvesv   ='Calculated LV end systolic volume (mL)'
    lvef_ev ='Calculated LV ejection fraction (echo vol est) (%)'
    lavol   ='Calculated Unscaled LA volume (mL)'
    la_ge6  ='LA size GE 6 cm'
    rvspnew ='Calculated right ventricular systolic pressure'
  ;

  drop poflag;

run;




*------------------------------------------------------------------------------;
* Create intraop multiple echos dataset                                        ;
*------------------------------------------------------------------------------;
data intraopechos1;
  set echo3;

  ioflag=0;
  if dt_surg=echo_dt and ((echotype in ('OREPI', 'ORTEE', 'ORTEEEPI', 'ORTEE3D', 'ORECHOPOST', 'ORECHOPRE')) )
  then ioflag=1;

  if ioflag=0 then delete;

  *** Echocardiographic derivatives;
  if lvidd  le lvisd then do;
    lvidd=.; lvisd=. ;
  end;

  *** LV Fractional Shortening;
  ifs=(lvidd-lvisd)/lvidd;

  *** LV Relative Wall Thickness - rhis is a measure of wall stress;
  irwt=2*pwt/lvidd;

  *******************************************************************************;
  * LV Mass                                                                      ;
  * Deveraux RB, Reichek N. Echocardiographic determination of left ventricular  ;
  * mass in man. Anatomic validation of the method. Circulation 1977,55:613-18   ;
  *                                                                              ;
  * Corrected for ASE method and overestimation                                  ;
  * Deveraux RB, Alonso DR, Lutas EM, Gottlieb GJ, Campo E, Sachs I, Reichek N.  ;
  * Echocardiographic assessment of left ventricular hypertrophy: Comparison to  ;
  * necropsy findings.  Am J Cardiol 1986,57:450-8                               ;
  *                                                                              ;
  * Indexing is controversial: straight BSA or BSA to a power (1.5)              ;
  * deSimone et al. J Am Coll Cardiol 1995,25:1056-62                            ;
  * Gutgesell HP. Growth of the human heart relative to body surface area.  Am J ;
  * Cardiol 1990,65:662-8                                                        ;
  *                                                                              ;
  * Units are: wall thickness in mm (millimeters), mass in g (grams)             ;
  *******************************************************************************;

  ilvmass=0.80*(1.04*(((lvidd + pwt + ivswt)**3) - lvidd**3)) + 0.6;
  *plvmassi=plvmass/bsa;
  *ms_lvmas=0; *if lvmassi=. then ms_lvmas=1;

  *******************************************************************************;
  * LV Volumes                                                                   ;
  * Teichholz LE, Kreulen T, Herman MI, Gorlin R.  Problems in                   ;
  * exhocardiocardiographic-angiographic correlations in the presence or absence ;
  * of asynergy. Am J Cardiol 1976,37:7-11                                       ;
  *                                                                              ;
  * Volumes are expressed in mL (milliliters) from dimensions in mm (millimeters);
  *                                                                              ;
  *******************************************************************************;

  ilvedv=(7.0/(2.4 + lvidd))*(lvidd**3);
  ilvesv=(7.0/(2.4 + lvisd))*(lvisd**3);
  ilvef_ev=100*(ilvedv-ilvesv)/ilvedv;
  *lvedvi=lvedv/bsa; *lvesvi=lvesv/bsa;

  *** LA Volume;
  *** No formula, but assume that it will be a scaled cubed function;
 pi=2*arcos(0);
  ilavol=(4*pi/3)*((ladia/2)**3);
  *lavoli=lavol/bsa;
  ila_ge6=(ladia ge 6);
  if ladia=. then ila_ge6=.;
  drop pi;

  *******************************************************************************;
  * RV systolic pressure                                                         ;
  * Yock PG, Popp RL. Noninvasive estimation of right ventricular systolic       ;
  * pressure by Doppler ultrasound in patients with tricuspid regurgitation.     ;
  * Circulation 1984,70:657-62.                                                  ;
  *                                                                              ;
  * From Bernoulli relation, RV pressure = RA pressure + 4*((TV velocity)^2),    ;
  * where TV velocity is peak tricuspid velocity in m/s (meters/second) and      ;
  * RV and RA pressures are measured in mmHg (millimetes of mercury)             ;
  *                                                                              ;
  * RA pressure may be estimated from jugular pressure or held constant. The     ;
  * latter is undesirable because PA pressure and RA pressure are positively     ;
  * correlated. Therefore:                                                       ;
  *                                                                              ;
  * Berger (JACC 1985,6:359-65) suggests a regression equation without RA        ;
  * pressure that multiplies this quantity by 1.23. Berger M, Haimowitz A,       ;
  * Van Tosh A, Berdoff RL, Goldberg E. Quantitative assessment of pulmonary     ;
  * hypertension in patients with tricuspid regurgitation using continuous wave  ;
  * Doppler ultrasound. J Am Coll Cardiol 1985,6:359-65.                         ;
  *                                                                              ;
  * By regression analysis, Wu used: RVSPNEW = 10 + 4*(TVPKVEL**2)               ;
  *******************************************************************************;

  *** RV systolic pressure extremes have few numbers;
  irvspnew=10 +4*((tv_rvel/100)**2);

  label
    ifs      ='Intra-op Calculated Fractional shortening'
    irwt     ='Intra-op Calculated LV relative wall thickness'
    ilvmass  ='Intra-op Calculated LV mass (ASE-cube mass, corrected)(g)'
    ilvedv   ='Intra-op Calculated LV end diatolic volume (mL)'
    ilvesv   ='Intra-op Calculated LV end systolic volume (mL)'
    ilvef_ev ='Intra-op Calculated LV ejection fraction (echo vol est) (%)'
    ilavol   ='Intra-op Calculated Unscaled LA volume (mL)'
    ila_ge6  ='Intra-op LA size GE 6 cm'
    irvspnew ='Intra-op Calculated right ventricular systolic pressure'
  ;

  label
    echo_plc='Intraop Echo location'
    echotype='Intraop Echo type'
    ladia   ='Intraop Left atrial diameter (cm)'
    lvidd   ='Intraop Left ventricular inner diameter in diastole (cm)'
    lvisd   ='Intraop Left ventricular inner diameter in systole (cm)'
    pwt     ='Intraop Posterior wall thickness (cm)'
    ivswt   ='Intraop Intraventricular septal thickness (cm)'
    ef_echo ='Intraop Ejection fraction (%)'
    av_regn ='Intraop AV regurgitation degree'
    av_stenn='Intraop AV stenosis degree'
    avpkgrad='Intraop AV peak gradient (mmHg)'
    avmngrad='Intraop AV mean gradient (mmHg)'
    av_area ='Intraop AV area (cm^2)'
    mv_regn ='Intraop MV regurgitation degree'
    mv_stenn='Intraop MV stenosis degree'
    mvpkgrad='Intraop MV peak gradient (mmHg)'
    mvmngrad='Intraop MV mean gradient (mmHg)'
    tv_regn ='Intraop TV regurgitation degree'
    tv_rvel ='Intraop TV regurgitation velocity'
    las_area='Intraop Left atrial systolic area (cm^2)'
    lad_area='Intraop Left atrial diastolic area (cm^2)'
    lvedvol ='Intraop Left ventricular end-diastolic volume(mL)'
    lvesvol ='Intraop Left ventricular end-systolic volume (mL)'
    rvsp    ='Intraop Right ventricular systolic Pressure'
    sys_funn='Intraop Left ventricular systolic function-1:NORM,2:MILD,3:MOD,4:MSEV,5:SEV'
    apicsept='Intraop Wall Motion: Apical Septum'
    apic_inf='Intraop Wall Motion: Apical Inferior'
    apic_lat='Intraop Wall Motion: Apical Lateral'
    apic_ant='Intraop Wall Motion: Apical Anterior'
    bas_ants='Intraop Wall Motion: Basal Anterior Septum'
    bas_sept='Intraop Wall Motion: Basal Septum'
    bas_infe='Intraop Wall Motion: Basal Inferior'
    bas_post='Intraop Wall Motion: Basal Posterior'
    bas_late='Intraop Wall Motion: Basal Lateral'
    bas_ante='Intraop Wall Motion: Basal Anterior'
    midantse='Intraop Wall Motion: Mid Anterior Septum'
    mid_sept='Intraop Wall Motion: Mid Septum'
    mid_infe='Intraop Wall Motion: Mid Inferior'
    mid_post='Intraop Wall Motion: Mid Posterior'
    mid_late='Intraop Wall Motion: Mid Lateral'
    mid_ante='Intraop Wall Motion: Mid Anterior'
    lv_mass ='Intraop LV Mass'
    lvotpkgr='Intraop LV Outflow Track Peak Gradient'
    lvotmngr='Intraop LV Outflow Track Mean Gradient'
    lvotpkga='Intraop LVOT Peak Gradient after Amyl Nitrite'
    lvotmnga='Intraop LVOT Mean Gradient after Amyl Nitrite'
    la_throm='Intraop Left Atrial Thrombosis'
    di_fun  ='Intraop Diastolic Function'
    di_funrc='Intraop Diastolic Function Res Con'
    di_funwm='Intraop Diastolic Function Filling Pattern w/ Maneuver'
    di_funm ='Intraop Diastolic Function Maneuver Type'
    di_funfp='Intraop Diastolic Function Filling Pressures'
    di_funcd='Intraop Dias Func Color Dopper m-Mode Flow Propagation'
    lvdpex  ='Intraop LV Dilation Post Exercise'
    sys_fcnp='Intraop Systolic Function Pattern'
    ras_area='Intraop Right Atrium Systolic Area'
    rad_area='Intraop Right Atrium Diastolic Area'
    rvs_area='Intraop Right Ventricle Systolic Area'
    rvd_area='Intraop Right Ventricle Diastolic Area'
    rvesvol ='Intraop Right Ventricle End Systolic Area'
    rvedvol ='Intraop Right Ventricle End Diastolic Area'
   ;

  rename
    dtm_echo=idtmecho
    echo_plc= iecho_pl
    echotype= iechotyp
    ladia   = iladia
    lvidd   = ilvidd
    lvisd   = ilvisd
    pwt     = ipwt
    ivswt   = iivswt
    ef_echo = ief_echo
    avpkgrad= iavpkgra
    avmngrad= iavmngra
    av_regn = iav_regn
    av_stenn= iav_sten
    av_area = iav_area
    mv_regn = imv_regn
    mv_stenn= imv_sten
    mvpkgrad= imvpkgra
    mvmngrad= imvmngra
    tv_regn = itv_regn
    tv_rvel = itv_rvel
    las_area= ilasarea
    lad_area= iladarea
    lvedvol = ilvedvol
    lvesvol = ilvesvol
    rvsp    = irvsp
    sys_funn= isys_fun
    apicsept= iapicsep
    apic_inf= iapic_in
    apic_lat= iapic_la
    apic_ant= iapic_an
    bas_ants= ibas_ans
    bas_sept= ibas_sep
    bas_infe= ibas_inf
    bas_post= ibas_pos
    bas_late= ibas_lat
    bas_ante= ibas_ane
    midantse= imidants
    mid_sept= imid_sep
    mid_infe= imid_inf
    mid_post= imid_pos
    mid_late= imid_lat
    mid_ante= imid_ant
    lv_mass = ilv_mass
    lvotpkgr= ilvotpgr
    lvotmngr= ilvotmgr
    lvotpkga= ilvotpga
    lvotmnga= ilvotmga
    la_throm= ilathrom
    di_fun  = idifun
    di_funrc= idifunrc
    di_funwm= idifunwm
    di_funm = idifunm
    di_funfp= idifunfp
    di_funcd= idifuncd
    lvdpex  = ilvdpex
    sys_fcnp= isysfcnp
    ras_area= irasarea
    rad_area= iradarea
    rvs_area= irvsarea
    rvd_area= irvdarea
    rvesvol = irvesvol
    rvedvol = irvedvol
  ;

  drop ioflag;
run;


*------------------------------------------------------------------------------;
* Create preop echos dataset                                                   ;
*------------------------------------------------------------------------------;
data preopecho1;
  set echo3;



  prflag=0;
  /* Per Dr. Blackstone, the echo types that are used for analysis should be limited to plain TTE only.
     In the past, TTE, SE, ARBUT and DOB types were used. (9/7/2011)

     1/9/2014
     Per Dr. Blackstone, the following 3 echo types are very specialized and we are not interested in them
     ECHOC   - Echo with contrast
     ECHOAV  - When a pt comes in with a BiV pacer, they don an AV Optimization echo
     PROECHO - Echo done during a procedure
  */

  if dtm_echo < dtn_inst and ((echotype in ('TTE', 'ECHO', 'XECHO' )))

  then prflag=1;

  if prflag=0 then delete;



  label
    echo_plc='Preop Echo location'
    echotype='Preop Echo type'
    ladia   ='Preop Left atrial diameter (cm)'
    lvidd   ='Preop Left ventricular inner diameter in diastole (cm)'
    lvisd   ='Preop Left ventricular inner diameter in systole (cm)'
    pwt     ='Preop Posterior wall thickness (cm)'
    ivswt   ='Preop Intraventricular septal thickness (cm)'
    ef_echo ='Preop Ejection fraction (%)'
    av_regn ='Preop AV regurgitation degree'
    av_stenn='Preop AV stenosis degree'
    avpkgrad='Preop AV peak gradient (mmHg)'
    avmngrad='Preop AV mean gradient (mmHg)'
    av_area ='Preop AV area (cm^2)'
    mv_regn ='Preop MV regurgitation degree'
    mv_stenn='Preop MV stenosis degree'
    mvpkgrad='Preop MV peak gradient (mmHg)'
    mvmngrad='Preop MV mean gradient (mmHg)'
    tv_regn ='Preop TV regurgitation degree'
    tv_rvel ='Preop TV regurgitation velocity'
    las_area='Preop Left atrial systolic area (cm^2)'
    lad_area='Preop Left atrial diastolic area (cm^2)'
    lvedvol ='Preop Left ventricular end-diastolic volume (mL)'
    lvesvol ='Preop Left ventricular end-systolic volume (mL)'
    rvsp    ='Preop Right ventricular systolic pressure'
    sys_funn='Preop Left ventricular systolic function-1:NORM,2:MILD,3:MOD,4:MSEV,5:SEV'
    aoszascm = 'Preop mid ascending aorta diameter'
    aosin_dm   = 'Preop aorta sinus diameter'
    apicsept='Preop Wall Motion: Apical Septum'
    apic_inf='Preop Wall Motion: Apical Inferior'
    apic_lat='Preop Wall Motion: Apical Lateral'
    apic_ant='Preop Wall Motion: Apical Anterior'
    bas_ants='Preop Wall Motion: Basal Anterior Septum'
    bas_sept='Preop Wall Motion: Basal Septum'
    bas_infe='Preop Wall Motion: Basal Inferior'
    bas_post='Preop Wall Motion: Basal Posterior'
    bas_late='Preop Wall Motion: Basal Lateral'
    bas_ante='Preop Wall Motion: Basal Anterior'
    midantse='Preop Wall Motion: Mid Anterior Septum'
    mid_sept='Preop Wall Motion: Mid Septum'
    mid_infe='Preop Wall Motion: Mid Inferior'
    mid_post='Preop Wall Motion: Mid Posterior'
    mid_late='Preop Wall Motion: Mid Lateral'
    mid_ante='Preop Wall Motion: Mid Anterior'
    lv_mass ='Preop LV Mass'
    lvotpkgr='Preop LV Outflow Track Peak Gradient'
    lvotmngr='Preop LV Outflow Track Mean Gradient'
    lvotpkga='Preop LVOT Peak Gradient after Amyl Nitrite'
    lvotmnga='Preop LVOT Mean Gradient after Amyl Nitrite'
    la_throm='Preop Left Atrial Thrombosis'
    di_fun  ='Preop Diastolic Function'
    di_funrc='Preop Diastolic Function Res Con'
    di_funwm='Preop Diastolic Function Filling Pattern w/ Maneuver'
    di_funm ='Preop Diastolic Function Maneuver Type'
    di_funfp='Preop Diastolic Function Filling Pressures'
    di_funcd='Preop Diastolic Func Color Dopper m-Mode Flow Propagation'
    lvdpex  ='Preop LV Dilation Post Exercise'
    sys_fcnp='Preop Systolic Function Pattern'
    ras_area='Preop Right Atrium Systolic Area'
    rad_area='Preop Right Atrium Diastolic Area'
    rvs_area='Preop Right Ventricle Systolic Area'
    rvd_area='Preop Right Ventricle Diastolic Area'
    rvesvol ='Preop Right Ventricle End Systolic Area'
    rvedvol ='Preop Right Ventricle End Diastolic Area'
  ;

  rename
    echo_dt = pecho_dt
    dtm_echo=pdtmecho
    echo_plc= pecho_pl
    echotype= pechotyp
    ladia   = pladia
    lvidd   = plvidd
    lvisd   = plvisd
    pwt     = ppwt
    ivswt   = pivswt
    ef_echo = pef_echo
    avpkgrad= pavpkgra
    avmngrad= pavmngra
    av_regn = pav_regn
    av_stenn= pav_sten
    av_area = pav_area
    mv_regn = pmv_regn
    mv_stenn= pmv_sten
    mvpkgrad= pmvpkgra
    mvmngrad= pmvmngra
    tv_regn = ptv_regn
    tv_rvel = ptv_rvel
    las_area= plasarea
    lad_area= pladarea
    lvedvol = plvedvol
    lvesvol = plvesvol
    rvsp    = prvsp
    sys_funn= psys_fun
    aoszascm =pmidaadm
    aosin_dm=paosindm
    apicsept= papicsep
    apic_inf= papic_in
    apic_lat= papic_la
    apic_ant= papic_an
    bas_ants= pbas_ans
    bas_sept= pbas_sep
    bas_infe= pbas_inf
    bas_post= pbas_pos
    bas_late= pbas_lat
    bas_ante= pbas_ane
    midantse= pmidants
    mid_sept= pmid_sep
    mid_infe= pmid_inf
    mid_post= pmid_pos
    mid_late= pmid_lat
    mid_ante= pmid_ant
    lv_mass = plv_mass
    lvotpkgr= plvotpgr
    lvotmngr= plvotmgr
    lvotpkga= plvotpga
    lvotmnga= plvotmga
    la_throm= plathrom
    di_fun  = pdifun
    di_funrc= pdifunrc
    di_funwm= pdifunwm
    di_funm = pdifunm
    di_funfp= pdifunfp
    di_funcd= pdifuncd
    lvdpex  = plvdpex
    sys_fcnp= psysfcnp
    ras_area= prasarea
    rad_area= pradarea
    rvs_area= prvsarea
    rvd_area= prvdarea
    rvesvol = prvesvol
    rvedvol = prvedvol
  ;

  drop prflag;
run;


proc sort data = preopecho1; by ccfid dt_surg pecho_dt; run;
data preopecho2;
  set preopecho1;
  by ccfid dt_surg pecho_dt;

  if first.dt_surg then do;
     pecho_n = 0;
  end;

  pecho_n = pecho_n+1;

  retain pecho_n;
run;


* Unique population by ccfid and surgery date;
proc sort data = preopecho2; by ccfid dtn_inst pecho_dt; run;
data population;
  set preopecho2;
  by ccfid dtn_inst pecho_dt;
  if last.dtn_inst;

  dt_lecho = pecho_dt;
  format dt_lecho mmddyy10.;

  label
   dt_lecho = 'Date of most recent echo prior surgery'
   pecho_n = 'Number of preop echos prior to surgery'
   ;
  keep ccfid pecho_n dtn_inst pecho_dt dt_lecho dt_surg;
run;


**********************************************************************************;
*This macro takes the most recent echo measurements prior to the CT and/or Surgery;
*Assumes intraop and post-op echos are already sorted out in above ECHO program   ;
**********************************************************************************;
%macro create(measure_name);

  *Keeps only the echo measure we specify (takes one measure at a time);
  data &measure_name._1;
    set preopecho2;
    keep ccfid dt_surg &measure_name pdtmecho;
  run;

  *Deletes records missing echo data;
  data &measure_name._2;
    set &measure_name._1;

    *This returns the variable type for the next data processing steps;
    x = vtype(&measure_name);

    *If a character value (denoted by 'C'), drops missing values;
    if x = 'C' then do;
      if &measure_name = ''  then  delete;
    end;

    *If a numeric value (denoted by 'N'), drops missing values;
    if x = 'N' then do;
      if &measure_name = .  then  delete;
    end;

    drop x;

  run;

  *Takes the last non-missing record for the specified echo measure;
  proc sort data = &measure_name._2; by ccfid dt_surg pdtmecho; run;
  data &measure_name._3;
    set &measure_name._2 ;
    by ccfid dt_surg pdtmecho;
    if last.dt_surg;
  run;

  *Renames the date variable so that we can trace the date of this echo
   measure back if we need to;
  data &measure_name;
    set &measure_name._3  (rename = (pdtmecho = &measure_name._dtm));
    *label &measure_name._dt = 'Date of &measure_name';
  run;

%mend create;

*Running the macro for the echo measures contained in the template;
 %create(pecho_pl) ;
 %create(pechotyp) ;
 %create(pladia)   ;
 %create(plvidd)   ;
 %create(plvisd)   ;
 %create(ppwt)     ;
 %create(pivswt)   ;
 %create(pef_echo) ;
 %create(pavpkgra) ;
 %create(pavmngra) ;
 %create(pav_regn) ;
 %create(pav_sten) ;
 %create(pav_area) ;
 %create(pmv_regn) ;
 %create(pmv_sten) ;
 %create(pmvpkgra) ;
 %create(pmvmngra) ;
 %create(ptv_regn) ;
 %create(ptv_rvel) ;
 %create(plasarea) ;
 %create(pladarea) ;
 %create(plvedvol) ;
 %create(plvesvol) ;
 %create(prvsp)    ;
 %create(psys_fun) ;
 %create(pmidaadm) ;
 %create(paosindm) ;
 %create(papicsep) ;
 %create(papic_in) ;
 %create(papic_la) ;
 %create(papic_an) ;
 %create(pbas_ans) ;
 %create(pbas_sep) ;
 %create(pbas_inf) ;
 %create(pbas_pos) ;
 %create(pbas_lat) ;
 %create(pbas_ane) ;
 %create(pmidants) ;
 %create(pmid_sep) ;
 %create(pmid_inf) ;
 %create(pmid_pos) ;
 %create(pmid_lat) ;
 %create(pmid_ant) ;
 %create(plv_mass) ;
 %create(plvotpgr) ;
 %create(plvotmgr) ;
 %create(plvotpga) ;
 %create(plvotmga) ;
 %create(plathrom) ;
 %create(pdifun)   ;
 %create(pdifunrc) ;
 %create(pdifunwm) ;
 %create(pdifunm)  ;
 %create(pdifunfp) ;
 %create(pdifuncd) ;
 %create(plvdpex)  ;
 %create(psysfcnp) ;
 %create(prasarea) ;
 %create(pradarea) ;
 %create(prvsarea) ;
 %create(prvdarea) ;
 %create(prvesvol) ;
 %create(prvedvol) ;


***************************************************************;
*Macro joining the various echo measurement datasets           ;
***************************************************************;

%macro joinecho(x1,x2,x3);
  proc sql;
    create table &x1 as
    select * from &x2 left join &x3 on
    &x2..ccfid=&x3..ccfid and &x2..dt_surg=&x3..dt_surg;
   quit;
%mend joinecho;

 %joinecho(all1, population, pecho_pl);
 %joinecho(all2, all1, pechotyp);
/* %joinecho(all3, all2, pechotyn);
 %joinecho(all4, all3, pechotim); */
 %joinecho(all5, all2, pladia);
 %joinecho(all6, all5, plvidd);
 %joinecho(all7, all6, plvisd);
 %joinecho(all8, all7, ppwt);
 %joinecho(all9, all8, pivswt);
 %joinecho(all10, all9, pef_echo);
 %joinecho(all11, all10, pavpkgra);
 %joinecho(all12, all11, pavmngra);
 %joinecho(all13, all12, pav_regn);
 %joinecho(all14, all13, pav_sten);
 %joinecho(all15, all14, pav_area);
 %joinecho(all16, all15, pmv_regn);
 %joinecho(all17, all16, pmv_sten);
 %joinecho(all18, all17, pmvpkgra);
 %joinecho(all19, all18, pmvmngra);
 %joinecho(all20, all19, ptv_regn);
 %joinecho(all21, all20, ptv_rvel);
 %joinecho(all22, all21, plasarea);
 %joinecho(all23, all22, pladarea);
 %joinecho(all24, all23, plvedvol);
 %joinecho(all25, all24, plvesvol);
 %joinecho(all26, all25, prvsp);
 %joinecho(all27, all26, psys_fun);
 %joinecho(all28, all27, pmidaadm);
 %joinecho(all29, all28, paosindm);
 %joinecho(all30, all29, papicsep);
 %joinecho(all31, all30, papic_in);
 %joinecho(all32, all31, papic_la);
 %joinecho(all33, all32, papic_an);
 %joinecho(all34, all33, pbas_ans);
 %joinecho(all35, all34, pbas_sep);
 %joinecho(all36, all35, pbas_inf);
 %joinecho(all37, all36, pbas_pos);
 %joinecho(all38, all37, pbas_lat);
 %joinecho(all39, all38, pbas_ane);
 %joinecho(all40, all39, pmidants);
 %joinecho(all41, all40, pmid_sep);
 %joinecho(all42, all41, pmid_inf);
 %joinecho(all43, all42, pmid_pos);
 %joinecho(all44, all43, pmid_lat);
 %joinecho(all45, all44, pmid_ant);
/* %joinecho(all46, all45, papindex); */
 %joinecho(all47, all45, plv_mass);
/* %joinecho(all48, all47, paoregfr); */
 %joinecho(all49, all47, plvotpgr);
 %joinecho(all50, all49, plvotmgr);
 %joinecho(all51, all50, plvotpga);
 %joinecho(all52, all51, plvotmga);
 %joinecho(all53, all52, plathrom);
/*%joinecho(all54, all53, plathloc);
 %joinecho(all55, all54, plathtyp);*/
 %joinecho(all56, all53, pdifun);
 /*%joinecho(all57, all56, pdifunre);
 %joinecho(all58, all57, pdifunco);*/
 %joinecho(all59, all56, pdifunrc);
 /*%joinecho(all60, all59, pdifuner);*/
 %joinecho(all61, all59, pdifunwm);
 %joinecho(all62, all61, pdifunm);
 %joinecho(all63, all62, pdifunfp);
 %joinecho(all64, all63, pdifuncd);
 %joinecho(all65, all64, plvdpex);
 /*%joinecho(all66, all65, plvvolc);
 %joinecho(all67, all66, plvvolpe);
 %joinecho(all68, all67, plvtestr);
 %joinecho(all69, all68, plvvasbi);
 %joinecho(all70, all69, plvvasbs);
 %joinecho(all71, all70, plvvasbv);
 %joinecho(all72, all71, ppretest);
 %joinecho(all73, all72, ppostest);*/
 %joinecho(all74, all65, psysfcnp);
 /*%joinecho(all75, all74, pttime);*/
 %joinecho(all76, all74, prasarea);
 %joinecho(all77, all76, pradarea);
 %joinecho(all78, all77, prvsarea);
 %joinecho(all79, all78, prvdarea);
 %joinecho(all80, all79, prvesvol);
 %joinecho(all81, all80, prvedvol);



*Joining number of preop echos (pecho_n) and last echo date back to datasets;
data preopecho4;
    set all81;
run;


*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@;
*Comment out the above code and run this section if you want to take the
 average of the available echo values;
%macro skip_old;
********************************************************************;
*Old echo section which takes the mean of most numeric echo measures;
********************************************************************;

***Calculate means for multiple preop echos;
proc sort data=preopecho2; by ccfid dtn_inst; run;
proc means data=preopecho2 mean noprint;
  var pladia plvidd plvisd ppwt pivswt pef_echo pav_regn pav_sten pavpkgra pavmngra
      pav_area ptv_regn pmv_regn pmv_sten psys_fun pmvpkgra pmvmngra ptv_rvel
      plasarea pladarea prvsp plvedvol plvesvol pmidaadm paosindm;
  by ccfid dtn_inst;
  output out=preopecho3a mean=;

run;

*** Create unique preop echo file by keeping some vars from last preop echo;
proc sort data=preopecho2; by ccfid dtn_inst pdtmecho; run;
data preopecho3b;
  set preopecho2 (rename = (pecho_dt = dt_lecho));
  by ccfid dtn_inst pdtmecho;

  keep ccfid dt_surg dt_lecho dtn_inst pdtmecho pecho_pl pechotyp pechotyn pechotim
       papic_in papic_la papic_an pbas_ans pbas_sep pbas_inf pbas_pos pbas_lat
       pbas_ane pmidants pmid_sep pmid_inf pmid_pos pmid_lat pmid_ant pecho_n;
  if last.dtn_inst;

run;

*** Join echovarmenas and preopecho into unique record file;
proc sql;
  create table preopecho4 as
  select * from preopecho3a as a left join preopecho3bas b
  on a.ccfid = b.ccfid and
  a.dtn_inst = b.dtn_inst;
quit;

%mend skip_old;
*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@;


*** Round means for averaged rank order values back to integers ;
data preopecho5;
  set preopecho4;


%macro skip;
* these below are not numeric for now;

  pav_reg=round(pav_regn,1);
  pav_ste=round(pav_sten,1);
  plv_sfun=round(psys_fun,1);
  ptv_reg=round(ptv_regn,1);
  pmv_reg=round(pmv_regn,1);
  pmv_ste=round(pmv_sten,1); /*Added by Jason Zhong on 11/19/2010*/

  label pmv_ste='Rounded Preop MV stenosis degree'
        pav_reg='Rounded preop AV regurgitation grade'
        pav_ste='Rounded preop AV stenosis'
        plv_sfun = 'Rounded Preop LV systolic function-1:NORM,2:MILD,3:MOD,4:MSEV,5:SEV'
        ptv_reg  ='Rounded preop TV reguritation grade'
        pmv_reg  ='Rounded preop MV regurgitation grade'
    ;


%mend skip;

  plv_sfun=round(psys_fun,1);

 label       plv_sfun = 'Rounded Preop LV systolic function-1:NORM,2:MILD,3:MOD,4:MSEV,5:SEV';

  /*drop pav_regn pav_sten psys_fun ptv_regn pmv_regn pmv_sten;*/

  *** Echocardiographic derivatives;
  if plvidd  le plvisd then do;
    plvidd=.; plvisd=. ;
  end;

  *** LV Fractional Shortening;
  pfs=(plvidd-plvisd)/plvidd;

  *** LV Relative Wall Thickness - this is a measure of wall stress;
  prwt=2*ppwt/plvidd;

  *******************************************************************************;
  * LV Mass                                                                      ;
  * Deveraux RB, Reichek N. Echocardiographic determination of left ventricular  ;
  * mass in man. Anatomic validation of the method. Circulation 1977,55:613-18   ;
  *                                                                              ;
  * Corrected for ASE method and overestimation                                  ;
  * Deveraux RB, Alonso DR, Lutas EM, Gottlieb GJ, Campo E, Sachs I, Reichek N.  ;
  * Echocardiographic assessment of left ventricular hypertrophy: Comparison to  ;
  * necropsy findings.  Am J Cardiol 1986,57:450-8                               ;
  *                                                                              ;
  * Indexing is controversial: straight BSA or BSA to a power (1.5)              ;
  * deSimone et al. J Am Coll Cardiol 1995,25:1056-62                            ;
  * Gutgesell HP. Growth of the human heart relative to body surface area.  Am J ;
  * Cardiol 1990,65:662-8                                                        ;
  *                                                                              ;
  * Units are: wall thickness in mm (millimeters), mass in g (grams)             ;
  *******************************************************************************;

  plvmass=0.80*(1.04*(((plvidd + ppwt + pivswt)**3) - plvidd**3)) + 0.6;
  *plvmassi=plvmass/bsa;
  *ms_lvmas=0; *if lvmassi=. then ms_lvmas=1;

  *******************************************************************************;
  * LV Volumes                                                                   ;
  * Teichholz LE, Kreulen T, Herman MI, Gorlin R.  Problems in                   ;
  * exhocardiocardiographic-angiographic correlations in the presence or absence ;
  * of asynergy. Am J Cardiol 1976,37:7-11                                       ;
  *                                                                              ;
  * Volumes are expressed in mL (milliliters) from dimensions in mm (millimeters);
  *******************************************************************************;

  plvedv=(7.0/(2.4 + plvidd))*(plvidd**3);
  plvesv=(7.0/(2.4 + plvisd))*(plvisd**3);
  plvef_ev=100*(plvedv-plvesv)/plvedv;
  *plvedvi=plvedv/bsa; *plvesvi=plvesv/bsa;

  *** LA Volume;
  ***No formula, but assume that it will be a scaled cubed function;
 pi=2*arcos(0);
  plavol=(4*pi/3)*((pladia/2)**3);
  *plavoli=plavol/bsa;
  pla_ge6=(pladia ge 6);
  if pladia=. then pla_ge6=.;
  drop pi;

  *******************************************************************************;
  * RV systolic pressure                                                         ;
  * Yock PG, Popp RL. Noninvasive estimation of right ventricular systolic       ;
  * pressure by Doppler ultrasound in patients with tricuspid regurgitation.     ;
  * Circulation 1984,70:657-62.                                                  ;
  *                                                                              ;
  * From Bernoulli relation, RV pressure = RA pressure + 4*((TV velocity)^2),    ;
  * where TV velocity is peak tricuspid velocity in m/s (meters/second) and      ;
  * RV and RA pressures are measured in mmHg (millimetes of mercury)             ;
  *                                                                              ;
  * RA pressure may be estimated from jugular pressure or held constant. The     ;
  * latter is undesirable because PA pressure and RA pressure are positively     ;
  * correlated. Therefore:                                                       ;
  *                                                                              ;
  * Berger (JACC 1985,6:359-65) suggests a regression equation without RA        ;
  * pressure that multiplies this quantity by 1.23. Berger M, Haimowitz A,       ;
  * Van Tosh A, Berdoff RL, Goldberg E. Quantitative assessment of pulmonary     ;
  * hypertension in patients with tricuspid regurgitation using continuous wave  ;
  * Doppler ultrasound. J Am Coll Cardiol 1985,6:359-65.                         ;
  *                                                                              ;
  * By regression analysis, Wu used: RVSPNEW = 10 + 4*(TVPKVEL**2)               ;
  *******************************************************************************;

  *** RV systolic pressure extremes have few numbers;
  prvspnew=10 +4*((ptv_rvel/100)**2);

  label
    pfs     ='Preop Calculated Fractional shortening'
    prwt    ='Preop Calculated LV relative wall thickness'
    plvmass ='Preop Calculated LV mass (ASE-cube mass, corrected) (g)'
    plvedv  ='Preop Calculated LV end diastolic volume (mL)'
    plvesv  ='Preop Calculated LV end systolic volume (mL)'
    plvef_ev='Preop Calculated LV ejection fraction (echo vol est) (%)'
    plavol  ='Preop Calculated Unscaled LA volume'
    pla_ge6 ='Preop LA size GE 6 cm'
    prvspnew='Preop Calculated right ventricular systolic pressure'
   /* pav_reg ='Preop AV regurgitation degree'
    pav_ste ='Preop AV stenosis degree'
    ptv_reg ='Preop TV regurgitation degree'
    pmv_reg ='Preop MV regurgitation degree'
    plv_sfun='Preop LV systolic function rank-1:NORM,2:MILD,3:MOD,4:MSEV,5:SEV' */
    pecho_n = 'Number of preop echos'
    dt_lecho = 'Date of most recent echo prior surgery';
  ;

run;










*______________________________________________________________________________;
*                                                                              ;
* Final Echo Datasets                                                          ;
*______________________________________________________________________________;
*                                                                              ;

%macro skip;

****Preop echo dataset with dates for individual preop measures;
data library.builtpreopecho_alldates;
  set preopecho5;
  *set preopecho5 preopecho5_o;
run;

%mend skip;


****Preop echo data without the dates for individual measures;
data library.builtpreopecho;
  set preopecho5;
  *set preopecho5 preopecho5_o;

 drop
  /*paoregfr_dtm  paosindm_dtm*/  papic_an_dtm  papic_in_dtm
  papic_la_dtm  papicsep_dtm  /*papindex_dtm*/  pav_area_dtm  pav_regn_dtm
  pav_sten_dtm  pavmngra_dtm  pavpkgra_dtm  pbas_ane_dtm  pbas_ans_dtm  pbas_inf_dtm
  pbas_lat_dtm  pbas_pos_dtm  pbas_sep_dtm  pdifun_dtm    pdifuncd_dtm  /*pdifunco_dtm
  pdifuner_dtm*/  pdifunfp_dtm  pdifunm_dtm   pdifunrc_dtm  /*pdifunre_dtm*/  pdifunwm_dtm
  pecho_pl_dtm  /*pechotim_dtm  pechotyn_dtm*/  pechotyp_dtm  pef_echo_dtm
  pivswt_dtm    pladarea_dtm  pladia_dtm    plasarea_dtm  /*plathloc_dtm*/  plathrom_dtm
  /*plathtyp_dtm*/  plv_mass_dtm  plvdpex_dtm   plvedvol_dtm
  plvesvol_dtm  plvidd_dtm    plvisd_dtm    plvotmga_dtm  plvotmgr_dtm  plvotpga_dtm
  plvotpgr_dtm  /*plvtestr_dtm  plvvasbi_dtm  plvvasbs_dtm  plvvasbv_dtm  plvvolc_dtm
  plvvolpe_dtm*/  pmid_ant_dtm  pmid_inf_dtm  pmid_lat_dtm  pmid_pos_dtm  pmid_sep_dtm
  pmidaadm_dtm  pmidants_dtm  pmv_regn_dtm  pmv_sten_dtm  pmvmngra_dtm
  pmvpkgra_dtm  /*ppostest_dtm  ppretest_dtm*/  ppwt_dtm      pradarea_dtm  prasarea_dtm
  prvdarea_dtm  prvedvol_dtm  prvesvol_dtm  prvsarea_dtm  prvsp_dtm     psys_fun_dtm
  psysfcnp_dtm  /*pttime_dtm*/    ptv_regn_dtm  ptv_rvel_dtm

/****Uncomment this section if you have old warehouse data;  */
  /*
  pav_area_dt   pav_regn_dt   pav_sten_dt   pavmngra_dt   pavpkgra_dt
  pecho_pl_dt   pechotim_dt   pechotyp_dt   pef_echo_dt   pivswt_dt     pladarea_dt
  pladia_dt     plasarea_dt   plvedvol_dt   plvesvol_dt   plvidd_dt     plvisd_dt
  pmidants_dt   pmv_regn_dt   pmv_sten_dt   pmvmngra_dt   pmvpkgra_dt
  ppwt_dt       prvsp_dt      psys_fun_dt   ptv_regn_dt   ptv_rvel_dt

  *this section should be left commented out if you do not have wall motion variables for the
   old cohort;
  papic_an_dt   papic_in_dt   papic_la_dt   papicsep_dt
  pbas_ane_dt   pbas_ans_dt   pbas_inf_dt   pbas_lat_dt   pbas_pos_dt   pbas_sep_dt
  pmidants_dt   pmid_ant_dt   pmid_inf_dt   pmid_lat_dt   pmid_pos_dt   pmid_sep_dt
  */

  ;

run;

title3 'Pre-op Echos';
proc contents data=library.builtpreopecho short;run;
proc means n nmiss mean std min max sum data=library.builtpreopecho;
run;



/*
***Data checks to see if pts are in the old AND the new data pulls
   Print patients below who overlap;
title 'Check for patients in both old and new datasets';
proc sql;
  create table test as
  select * from preopecho5 as a inner join preopecho5_o as b
  on a.ccfid = b.ccfid and
     a.dt_surg =b.dt_surg;
quit;

proc print data = preopecho5;
  var ccfid dt_surg echo_dt ;
  where ccfid in (XXXXXXXXXXXX);
run;
*/


%macro skip;
*these echo datasets are not needed at this point;
%mend skip;

****Multi-rowed postop echo dataset;
data library.postopechos;
  set postopechos1;
  *set postopechos1 postopechos1_o;
run;

title3 'Post-op Echos';
proc contents data=library.postopechos;
proc means n nmiss mean std min max sum data=library.postopechos;
run;


****Multi-rowed intraop echo dataset;
data library.intraopechos;
  set intraopechos1;
  *set intraopechos1 intraopechos1_o;
run;

title3 'Intra-op Echos';
proc contents data=library.intraopechos;
proc means n nmiss mean std min max sum data=library.intraopechos;
run;





