library(tibble)

# Custom class
variable_label <- function(x, label) {
  structure(x, label = label, class = c("variable_label", class(x)))
}

# Custom print method
print.variable_label <- function(x, ...) {
  cat(label(x), "\n")
  NextMethod()
}

# Adding labels
newdata$ccfid <- variable_label(newdata$ccfid, 'CCF ID')
newdata$ladia <- variable_label(newdata$ladia, 'Left atrial diameter (cm)')
newdata$lvidd <- variable_label(newdata$lvidd, 'Left ventricular inner diameter in diastole (cm)')
newdata$lvisd <- variable_label(newdata$lvisd, 'Left ventricular inner diameter in systole (cm)')
newdata$pwt <- variable_label(newdata$pwt, 'Posterior wall thickness (cm)')
newdata$ivswt <- variable_label(newdata$ivswt, 'Intraventricular septal thickness (cm)')
newdata$ef_echo <- variable_label(newdata$ef_echo, 'Ejection fraction (%)')

newdata$avpkgrad <- variable_label(newdata$avpkgrad, 'AV peak gradient (mmHg)')
newdata$avmngrad <- variable_label(newdata$avmngrad, 'AV mean gradient (mmHg)')
newdata$av_area <- variable_label(newdata$av_area, 'AV area (cm^2)')


newdata$mvpkgrad <- variable_label(newdata$mvpkgrad, 'MV peak gradient (mmHg)')
newdata$mvmngrad <- variable_label(newdata$mvmngrad, 'MV mean gradient (mmHg)')

newdata$tv_rvel <- variable_label(newdata$tv_rvel, 'TV regurgitation velocity')
newdata$las_area <- variable_label(newdata$las_area, 'Left atrial systolic area (cm^2)')
newdata$lad_area <- variable_label(newdata$lad_area, 'Left atrial diastolic area (cm^2)')
newdata$lvedvol <- variable_label(newdata$lvedvol, 'Left ventricular end-diastolic volume (mL)')
newdata$lvesvol <- variable_label(newdata$lvesvol, 'Left ventricular end-systolic volume (mL)')
newdata$rvsp <- variable_label(newdata$rvsp, 'Right ventricular systolic pressure')

newdata$apicsept <- variable_label(newdata$apicsept, 'Wall Motion: Apical Septum')
newdata$apic_inf <- variable_label(newdata$apic_inf, 'Wall Motion: Apical Inferior')
newdata$apic_lat <- variable_label(newdata$apic_lat, 'Wall Motion: Apical Lateral')
newdata$apic_ant <- variable_label(newdata$apic_ant, 'Wall Motion: Apical Anterior')
newdata$bas_ants <- variable_label(newdata$bas_ants, 'Wall Motion: Basal Anterior Septum')
newdata$bas_sept <- variable_label(newdata$bas_sept, 'Wall Motion: Basal Septum')
newdata$bas_infe <- variable_label(newdata$bas_infe, 'Wall Motion: Basal Inferior')
newdata$bas_post <- variable_label(newdata$bas_post, 'Wall Motion: Basal Posterior')
newdata$bas_late <- variable_label(newdata$bas_late, 'Wall Motion: Basal Lateral')
newdata$bas_ante <- variable_label(newdata$bas_ante, 'Wall Motion: Basal Anterior')
newdata$midantse <- variable_label(newdata$midantse, 'Wall Motion: Mid Anterior Septum')
newdata$mid_sept <- variable_label(newdata$mid_sept, 'Wall Motion: Mid Septum')
newdata$mid_infe <- variable_label(newdata$mid_infe, 'Wall Motion: Mid Inferior')
newdata$mid_post <- variable_label(newdata$mid_post, 'Wall Motion: Mid Posterior')
newdata$mid_late <- variable_label(newdata$mid_late, 'Wall Motion: Mid Lateral')
newdata$mid_ante <- variable_label(newdata$mid_ante, 'Wall Motion: Mid Anterior')
newdata$lv_mass <- variable_label(newdata$lv_mass, 'LV Mass')
newdata$lvotpkgr <- variable_label(newdata$lvotpkgr, 'LV Outflow Track Peak Gradient')



newdata$di_fun <- variable_label(newdata$di_fun, 'Diastolic Function')

newdata$lvdpex <- variable_label(newdata$lvdpex, 'LV Dilation Post Exercise')
newdata$sys_fcnp <- variable_label(newdata$sys_fcnp, 'Systolic Function Pattern')
newdata$ras_area <- variable_label(newdata$ras_area, 'Right Atrium Systolic Area')

