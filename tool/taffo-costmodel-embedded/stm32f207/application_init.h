/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : application_init.h
 * @brief          : Header for application_init.c file.
 *                   This file contains the common defines of the application.
 ******************************************************************************
 * @attention
 *
 * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
 * All rights reserved.</center></h2>
 *
 * This software component is licensed by ST under BSD 3-Clause license,
 * the "License"; You may not use this file except in compliance with the
 * License. You may obtain a copy of the License at:
 *                        opensource.org/licenses/BSD-3-Clause
 *
 ******************************************************************************
 */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef APPLICATION_INIT
#define APPLICATION_INIT

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f2xx_hal.h"

  /* Private includes ----------------------------------------------------------*/
  /* USER CODE BEGIN Includes */

  /* USER CODE END Includes */

  /* Exported types ------------------------------------------------------------*/
  /* USER CODE BEGIN ET */

  /* USER CODE END ET */

  /* Exported constants --------------------------------------------------------*/
  /* USER CODE BEGIN EC */
  extern UART_HandleTypeDef huart3;
  /* USER CODE END EC */

  /* Exported macro ------------------------------------------------------------*/
  /* USER CODE BEGIN EM */

  /* USER CODE END EM */

  /* Exported functions prototypes ---------------------------------------------*/
  void Error_Handler(void);

  /* USER CODE BEGIN EFP */
  extern void main(void);
/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define A19_Pin GPIO_PIN_3
#define A19_GPIO_Port GPIOE
#define TRACE_CLK_Pin GPIO_PIN_2
#define TRACE_CLK_GPIO_Port GPIOE
#define FSMC_NBL1_Pin GPIO_PIN_1
#define FSMC_NBL1_GPIO_Port GPIOE
#define FSMC_NBL0_Pin GPIO_PIN_0
#define FSMC_NBL0_GPIO_Port GPIOE
#define MII_TXD3_Pin GPIO_PIN_8
#define MII_TXD3_GPIO_Port GPIOB
#define ULPI_D7_Pin GPIO_PIN_5
#define ULPI_D7_GPIO_Port GPIOB
#define MII_TXD1_Pin GPIO_PIN_14
#define MII_TXD1_GPIO_Port GPIOG
#define MII_TXD0_Pin GPIO_PIN_13
#define MII_TXD0_GPIO_Port GPIOG
#define TRST_Pin GPIO_PIN_4
#define TRST_GPIO_Port GPIOB
#define TDO_SWO_Pin GPIO_PIN_3
#define TDO_SWO_GPIO_Port GPIOB
#define FSMC_NE1_Pin GPIO_PIN_7
#define FSMC_NE1_GPIO_Port GPIOD
#define MicroSDCard_CLK_Pin GPIO_PIN_12
#define MicroSDCard_CLK_GPIO_Port GPIOC
#define TDI_Pin GPIO_PIN_15
#define TDI_GPIO_Port GPIOA
#define TCK_SWCLK_Pin GPIO_PIN_14
#define TCK_SWCLK_GPIO_Port GPIOA
#define TMS_SWDIO_Pin GPIO_PIN_13
#define TMS_SWDIO_GPIO_Port GPIOA
#define A20_Pin GPIO_PIN_4
#define A20_GPIO_Port GPIOE
#define TRACE_D2_Pin GPIO_PIN_5
#define TRACE_D2_GPIO_Port GPIOE
#define TRACE_D3_Pin GPIO_PIN_6
#define TRACE_D3_GPIO_Port GPIOE
#define I2C1_SDA_Pin GPIO_PIN_9
#define I2C1_SDA_GPIO_Port GPIOB
#define FSMC_NL_Pin GPIO_PIN_7
#define FSMC_NL_GPIO_Port GPIOB
#define I2C1_SCL_Pin GPIO_PIN_6
#define I2C1_SCL_GPIO_Port GPIOB
#define User_Button__Button_B4__Pin GPIO_PIN_15
#define User_Button__Button_B4__GPIO_Port GPIOG
#define SmartCard_CMDVCC_Pin GPIO_PIN_12
#define SmartCard_CMDVCC_GPIO_Port GPIOG
#define MII_TX_EN_Pin GPIO_PIN_11
#define MII_TX_EN_GPIO_Port GPIOG
#define FSMC_NE3_Pin GPIO_PIN_10
#define FSMC_NE3_GPIO_Port GPIOG
#define FSMC_NWAIT_Pin GPIO_PIN_6
#define FSMC_NWAIT_GPIO_Port GPIOD
#define D2_Pin GPIO_PIN_0
#define D2_GPIO_Port GPIOD
#define RS232_RX_Pin GPIO_PIN_11
#define RS232_RX_GPIO_Port GPIOC
#define RS232_TX_Pin GPIO_PIN_10
#define RS232_TX_GPIO_Port GPIOC
#define USB_FS_DP_Pin GPIO_PIN_12
#define USB_FS_DP_GPIO_Port GPIOA
#define DCMI_D7_Pin GPIO_PIN_7
#define DCMI_D7_GPIO_Port GPIOI
#define DCMI_D6_Pin GPIO_PIN_6
#define DCMI_D6_GPIO_Port GPIOI
#define DCMI_VSYNC_Pin GPIO_PIN_5
#define DCMI_VSYNC_GPIO_Port GPIOI
#define FSMC_NE2_Pin GPIO_PIN_9
#define FSMC_NE2_GPIO_Port GPIOG
#define FSMC_NWE_Pin GPIO_PIN_5
#define FSMC_NWE_GPIO_Port GPIOD
#define D3_Pin GPIO_PIN_1
#define D3_GPIO_Port GPIOD
#define I2S_SD_Pin GPIO_PIN_3
#define I2S_SD_GPIO_Port GPIOI
#define IO_Expander_INT_Pin GPIO_PIN_2
#define IO_Expander_INT_GPIO_Port GPIOI
#define USB_FS_DM_Pin GPIO_PIN_11
#define USB_FS_DM_GPIO_Port GPIOA
#define Anti_Tamper_Pin GPIO_PIN_13
#define Anti_Tamper_GPIO_Port GPIOC
#define LED3_Pin GPIO_PIN_9
#define LED3_GPIO_Port GPIOI
#define DCMI_D5_Pin GPIO_PIN_4
#define DCMI_D5_GPIO_Port GPIOI
#define FSMC_NOE_Pin GPIO_PIN_4
#define FSMC_NOE_GPIO_Port GPIOD
#define FSMC_CLK_Pin GPIO_PIN_3
#define FSMC_CLK_GPIO_Port GPIOD
#define MicroSDCard_CMD_Pin GPIO_PIN_2
#define MicroSDCard_CMD_GPIO_Port GPIOD
#define SmartCard_3_5V_Pin GPIO_PIN_15
#define SmartCard_3_5V_GPIO_Port GPIOH
#define I2S_SCLK_Pin GPIO_PIN_1
#define I2S_SCLK_GPIO_Port GPIOI
#define USB_FS_ID_Pin GPIO_PIN_10
#define USB_FS_ID_GPIO_Port GPIOA
#define PC14_OSC32_IN_Pin GPIO_PIN_14
#define PC14_OSC32_IN_GPIO_Port GPIOC
#define A0_Pin GPIO_PIN_0
#define A0_GPIO_Port GPIOF
#define MII_RX_ER_Pin GPIO_PIN_10
#define MII_RX_ER_GPIO_Port GPIOI
#define ULPI_DIR_Pin GPIO_PIN_11
#define ULPI_DIR_GPIO_Port GPIOI
#define MicroSDCard_Detect_Pin GPIO_PIN_13
#define MicroSDCard_Detect_GPIO_Port GPIOH
#define DCMI_D4_Pin GPIO_PIN_14
#define DCMI_D4_GPIO_Port GPIOH
#define I2S_WS_Pin GPIO_PIN_0
#define I2S_WS_GPIO_Port GPIOI
#define VBUS_FS_Pin GPIO_PIN_9
#define VBUS_FS_GPIO_Port GPIOA
#define PC15_OSC32_OUT_Pin GPIO_PIN_15
#define PC15_OSC32_OUT_GPIO_Port GPIOC
#define MII_CRS_Pin GPIO_PIN_2
#define MII_CRS_GPIO_Port GPIOH
#define MicroSDCard_D1_Pin GPIO_PIN_9
#define MicroSDCard_D1_GPIO_Port GPIOC
#define MCO_Pin GPIO_PIN_8
#define MCO_GPIO_Port GPIOA
#define PH0_OSC_IN_Pin GPIO_PIN_0
#define PH0_OSC_IN_GPIO_Port GPIOH
#define MII_COL_Pin GPIO_PIN_3
#define MII_COL_GPIO_Port GPIOH
#define MicroSDCard_D0_Pin GPIO_PIN_8
#define MicroSDCard_D0_GPIO_Port GPIOC
#define LED4_Pin GPIO_PIN_7
#define LED4_GPIO_Port GPIOC
#define PH1_OSC_OUT_Pin GPIO_PIN_1
#define PH1_OSC_OUT_GPIO_Port GPIOH
#define A2_Pin GPIO_PIN_2
#define A2_GPIO_Port GPIOF
#define A1_Pin GPIO_PIN_1
#define A1_GPIO_Port GPIOF
#define ULPI_NXT_Pin GPIO_PIN_4
#define ULPI_NXT_GPIO_Port GPIOH
#define LED2_Pin GPIO_PIN_8
#define LED2_GPIO_Port GPIOG
#define I2S_MCK_Pin GPIO_PIN_6
#define I2S_MCK_GPIO_Port GPIOC
#define A3_Pin GPIO_PIN_3
#define A3_GPIO_Port GPIOF
#define A4_Pin GPIO_PIN_4
#define A4_GPIO_Port GPIOF
#define OTG_FS_PowerSwitchOn_Pin GPIO_PIN_5
#define OTG_FS_PowerSwitchOn_GPIO_Port GPIOH
#define SmartCard_CLK_Pin GPIO_PIN_7
#define SmartCard_CLK_GPIO_Port GPIOG
#define LED1_Pin GPIO_PIN_6
#define LED1_GPIO_Port GPIOG
#define SmartCard_RST_Pin GPIO_PIN_7
#define SmartCard_RST_GPIO_Port GPIOF
#define SmartCard_OFF_Pin GPIO_PIN_6
#define SmartCard_OFF_GPIO_Port GPIOF
#define A5_Pin GPIO_PIN_5
#define A5_GPIO_Port GPIOF
#define DCMI_D3_Pin GPIO_PIN_12
#define DCMI_D3_GPIO_Port GPIOH
#define A15_Pin GPIO_PIN_5
#define A15_GPIO_Port GPIOG
#define A14_Pin GPIO_PIN_4
#define A14_GPIO_Port GPIOG
#define A13_Pin GPIO_PIN_3
#define A13_GPIO_Port GPIOG
#define DCMI_D2_Pin GPIO_PIN_11
#define DCMI_D2_GPIO_Port GPIOH
#define DCMI_D1_Pin GPIO_PIN_10
#define DCMI_D1_GPIO_Port GPIOH
#define D1_Pin GPIO_PIN_15
#define D1_GPIO_Port GPIOD
#define A12_Pin GPIO_PIN_2
#define A12_GPIO_Port GPIOG
#define ULPI_STP_Pin GPIO_PIN_0
#define ULPI_STP_GPIO_Port GPIOC
#define MII_MDC_Pin GPIO_PIN_1
#define MII_MDC_GPIO_Port GPIOC
#define MII_TXD2_Pin GPIO_PIN_2
#define MII_TXD2_GPIO_Port GPIOC
#define MII_TX_CLK_Pin GPIO_PIN_3
#define MII_TX_CLK_GPIO_Port GPIOC
#define SW1_Pin GPIO_PIN_2
#define SW1_GPIO_Port GPIOB
#define A11_Pin GPIO_PIN_1
#define A11_GPIO_Port GPIOG
#define MII_RXD2_Pin GPIO_PIN_6
#define MII_RXD2_GPIO_Port GPIOH
#define DCMI_HSYNC_Pin GPIO_PIN_8
#define DCMI_HSYNC_GPIO_Port GPIOH
#define DCMI_D0_Pin GPIO_PIN_9
#define DCMI_D0_GPIO_Port GPIOH
#define D0_Pin GPIO_PIN_14
#define D0_GPIO_Port GPIOD
#define A18_Pin GPIO_PIN_13
#define A18_GPIO_Port GPIOD
#define MII_RX_CLK_RMII_REF_CLK_Pin GPIO_PIN_1
#define MII_RX_CLK_RMII_REF_CLK_GPIO_Port GPIOA
#define WAKEUP_Pin GPIO_PIN_0
#define WAKEUP_GPIO_Port GPIOA
#define Audio_DAC_OUT_Pin GPIO_PIN_4
#define Audio_DAC_OUT_GPIO_Port GPIOA
#define MII_RXD0_Pin GPIO_PIN_4
#define MII_RXD0_GPIO_Port GPIOC
#define A7_Pin GPIO_PIN_13
#define A7_GPIO_Port GPIOF
#define A10_Pin GPIO_PIN_0
#define A10_GPIO_Port GPIOG
#define D10_Pin GPIO_PIN_13
#define D10_GPIO_Port GPIOE
#define MII_RXD3_Pin GPIO_PIN_7
#define MII_RXD3_GPIO_Port GPIOH
#define A17_Pin GPIO_PIN_12
#define A17_GPIO_Port GPIOD
#define A16_Pin GPIO_PIN_11
#define A16_GPIO_Port GPIOD
#define D15_Pin GPIO_PIN_10
#define D15_GPIO_Port GPIOD
#define MII_MDIO_Pin GPIO_PIN_2
#define MII_MDIO_GPIO_Port GPIOA
#define DCMI_PIXCK_Pin GPIO_PIN_6
#define DCMI_PIXCK_GPIO_Port GPIOA
#define ULPI_CLK_Pin GPIO_PIN_5
#define ULPI_CLK_GPIO_Port GPIOA
#define MII_RXD1_Pin GPIO_PIN_5
#define MII_RXD1_GPIO_Port GPIOC
#define A6_Pin GPIO_PIN_12
#define A6_GPIO_Port GPIOF
#define A9_Pin GPIO_PIN_15
#define A9_GPIO_Port GPIOF
#define D5_Pin GPIO_PIN_8
#define D5_GPIO_Port GPIOE
#define D6_Pin GPIO_PIN_9
#define D6_GPIO_Port GPIOE
#define D8_Pin GPIO_PIN_11
#define D8_GPIO_Port GPIOE
#define D11_Pin GPIO_PIN_14
#define D11_GPIO_Port GPIOE
#define ULPI_D5_Pin GPIO_PIN_12
#define ULPI_D5_GPIO_Port GPIOB
#define ULPI_D6_Pin GPIO_PIN_13
#define ULPI_D6_GPIO_Port GPIOB
#define D14_Pin GPIO_PIN_9
#define D14_GPIO_Port GPIOD
#define D13_Pin GPIO_PIN_8
#define D13_GPIO_Port GPIOD
#define ULPI_D0_Pin GPIO_PIN_3
#define ULPI_D0_GPIO_Port GPIOA
#define MII_RX_DV_RMII_CRSDV_Pin GPIO_PIN_7
#define MII_RX_DV_RMII_CRSDV_GPIO_Port GPIOA
#define ULPI_D2_Pin GPIO_PIN_1
#define ULPI_D2_GPIO_Port GPIOB
#define ULPI_D1_Pin GPIO_PIN_0
#define ULPI_D1_GPIO_Port GPIOB
#define OTG_FS_OverCurrent_Pin GPIO_PIN_11
#define OTG_FS_OverCurrent_GPIO_Port GPIOF
#define A8_Pin GPIO_PIN_14
#define A8_GPIO_Port GPIOF
#define D4_Pin GPIO_PIN_7
#define D4_GPIO_Port GPIOE
#define D7_Pin GPIO_PIN_10
#define D7_GPIO_Port GPIOE
#define D9_Pin GPIO_PIN_12
#define D9_GPIO_Port GPIOE
#define D12_Pin GPIO_PIN_15
#define D12_GPIO_Port GPIOE
#define ULPI_D3_Pin GPIO_PIN_10
#define ULPI_D3_GPIO_Port GPIOB
#define ULPI_D4_Pin GPIO_PIN_11
#define ULPI_D4_GPIO_Port GPIOB
#define MII_INT_Pin GPIO_PIN_14
#define MII_INT_GPIO_Port GPIOB
#define OneNAND_INT_Pin GPIO_PIN_15
#define OneNAND_INT_GPIO_Port GPIOB
  /* USER CODE BEGIN Private defines */

  /* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* APPLICATION_INIT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
