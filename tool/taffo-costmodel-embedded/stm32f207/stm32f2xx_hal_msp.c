/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file         stm32f2xx_hal_msp.c
 * @brief        This file provides code for the MSP Initialization
 *               and de-Initialization codes.
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

/* Includes ------------------------------------------------------------------*/
#include "application_init.h"
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN Define */

/* USER CODE END Define */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN Macro */

/* USER CODE END Macro */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* External functions --------------------------------------------------------*/
/* USER CODE BEGIN ExternalFunctions */

/* USER CODE END ExternalFunctions */

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */
/**
 * Initializes the Global MSP.
 */
void HAL_MspInit(void) {
  /* USER CODE BEGIN MspInit 0 */

  /* USER CODE END MspInit 0 */

  __HAL_RCC_SYSCFG_CLK_ENABLE();
  __HAL_RCC_PWR_CLK_ENABLE();

  HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_0);

  /* System interrupt init*/

  /* USER CODE BEGIN MspInit 1 */

  /* USER CODE END MspInit 1 */
}

/**
 * @brief RTC MSP Initialization
 * This function configures the hardware resources used in this example
 * @param hrtc: RTC handle pointer
 * @retval None
 */
void HAL_RTC_MspInit(RTC_HandleTypeDef* hrtc) {
  if (hrtc->Instance == RTC) {
    /* USER CODE BEGIN RTC_MspInit 0 */

    /* USER CODE END RTC_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_RTC_ENABLE();
    /* USER CODE BEGIN RTC_MspInit 1 */

    /* USER CODE END RTC_MspInit 1 */
  }
}

/**
 * @brief RTC MSP De-Initialization
 * This function freeze the hardware resources used in this example
 * @param hrtc: RTC handle pointer
 * @retval None
 */
void HAL_RTC_MspDeInit(RTC_HandleTypeDef* hrtc) {
  if (hrtc->Instance == RTC) {
    /* USER CODE BEGIN RTC_MspDeInit 0 */

    /* USER CODE END RTC_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_RTC_DISABLE();
    /* USER CODE BEGIN RTC_MspDeInit 1 */

    /* USER CODE END RTC_MspDeInit 1 */
  }
}

/**
 * @brief SD MSP Initialization
 * This function configures the hardware resources used in this example
 * @param hsd: SD handle pointer
 * @retval None
 */
void HAL_SD_MspInit(SD_HandleTypeDef* hsd) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if (hsd->Instance == SDIO) {
    /* USER CODE BEGIN SDIO_MspInit 0 */

    /* USER CODE END SDIO_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_SDIO_CLK_ENABLE();

    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    /**SDIO GPIO Configuration
    PC12     ------> SDIO_CK
    PD2     ------> SDIO_CMD
    PC9     ------> SDIO_D1
    PC8     ------> SDIO_D0
    */
    GPIO_InitStruct.Pin = MicroSDCard_CLK_Pin | MicroSDCard_D1_Pin | MicroSDCard_D0_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF12_SDIO;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    GPIO_InitStruct.Pin = MicroSDCard_CMD_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF12_SDIO;
    HAL_GPIO_Init(MicroSDCard_CMD_GPIO_Port, &GPIO_InitStruct);

    /* USER CODE BEGIN SDIO_MspInit 1 */

    /* USER CODE END SDIO_MspInit 1 */
  }
}

/**
 * @brief SD MSP De-Initialization
 * This function freeze the hardware resources used in this example
 * @param hsd: SD handle pointer
 * @retval None
 */
void HAL_SD_MspDeInit(SD_HandleTypeDef* hsd) {
  if (hsd->Instance == SDIO) {
    /* USER CODE BEGIN SDIO_MspDeInit 0 */

    /* USER CODE END SDIO_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_SDIO_CLK_DISABLE();

    /**SDIO GPIO Configuration
    PC12     ------> SDIO_CK
    PD2     ------> SDIO_CMD
    PC9     ------> SDIO_D1
    PC8     ------> SDIO_D0
    */
    HAL_GPIO_DeInit(GPIOC, MicroSDCard_CLK_Pin | MicroSDCard_D1_Pin | MicroSDCard_D0_Pin);

    HAL_GPIO_DeInit(MicroSDCard_CMD_GPIO_Port, MicroSDCard_CMD_Pin);

    /* USER CODE BEGIN SDIO_MspDeInit 1 */

    /* USER CODE END SDIO_MspDeInit 1 */
  }
}

/**
 * @brief UART MSP Initialization
 * This function configures the hardware resources used in this example
 * @param huart: UART handle pointer
 * @retval None
 */
void HAL_UART_MspInit(UART_HandleTypeDef* huart) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if (huart->Instance == USART3) {
    /* USER CODE BEGIN USART3_MspInit 0 */

    /* USER CODE END USART3_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_USART3_CLK_ENABLE();

    __HAL_RCC_GPIOC_CLK_ENABLE();
    /**USART3 GPIO Configuration
    PC11     ------> USART3_RX
    PC10     ------> USART3_TX
    */
    GPIO_InitStruct.Pin = RS232_RX_Pin | RS232_TX_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART3;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    /* USER CODE BEGIN USART3_MspInit 1 */

    /* USER CODE END USART3_MspInit 1 */
  }
}

/**
 * @brief UART MSP De-Initialization
 * This function freeze the hardware resources used in this example
 * @param huart: UART handle pointer
 * @retval None
 */
void HAL_UART_MspDeInit(UART_HandleTypeDef* huart) {
  if (huart->Instance == USART3) {
    /* USER CODE BEGIN USART3_MspDeInit 0 */

    /* USER CODE END USART3_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_USART3_CLK_DISABLE();

    /**USART3 GPIO Configuration
    PC11     ------> USART3_RX
    PC10     ------> USART3_TX
    */
    HAL_GPIO_DeInit(GPIOC, RS232_RX_Pin | RS232_TX_Pin);

    /* USER CODE BEGIN USART3_MspDeInit 1 */

    /* USER CODE END USART3_MspDeInit 1 */
  }
}

static uint32_t FSMC_Initialized = 0;

static void HAL_FSMC_MspInit(void) {
  /* USER CODE BEGIN FSMC_MspInit 0 */

  /* USER CODE END FSMC_MspInit 0 */
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if (FSMC_Initialized)
    return;
  FSMC_Initialized = 1;

  /* Peripheral clock enable */
  __HAL_RCC_FSMC_CLK_ENABLE();

  /** FSMC GPIO Configuration
  PE3   ------> FSMC_A19
  PE1   ------> FSMC_NBL1
  PE0   ------> FSMC_NBL0
  PD7   ------> FSMC_NE1
  PE4   ------> FSMC_A20
  PB7   ------> FSMC_NL
  PG10   ------> FSMC_NE3
  PD6   ------> FSMC_NWAIT
  PD0   ------> FSMC_D2
  PG9   ------> FSMC_NE2
  PD5   ------> FSMC_NWE
  PD1   ------> FSMC_D3
  PD4   ------> FSMC_NOE
  PD3   ------> FSMC_CLK
  PF0   ------> FSMC_A0
  PF2   ------> FSMC_A2
  PF1   ------> FSMC_A1
  PF3   ------> FSMC_A3
  PF4   ------> FSMC_A4
  PF5   ------> FSMC_A5
  PG5   ------> FSMC_A15
  PG4   ------> FSMC_A14
  PG3   ------> FSMC_A13
  PD15   ------> FSMC_D1
  PG2   ------> FSMC_A12
  PG1   ------> FSMC_A11
  PD14   ------> FSMC_D0
  PD13   ------> FSMC_A18
  PF13   ------> FSMC_A7
  PG0   ------> FSMC_A10
  PE13   ------> FSMC_D10
  PD12   ------> FSMC_A17
  PD11   ------> FSMC_A16
  PD10   ------> FSMC_D15
  PF12   ------> FSMC_A6
  PF15   ------> FSMC_A9
  PE8   ------> FSMC_D5
  PE9   ------> FSMC_D6
  PE11   ------> FSMC_D8
  PE14   ------> FSMC_D11
  PD9   ------> FSMC_D14
  PD8   ------> FSMC_D13
  PF14   ------> FSMC_A8
  PE7   ------> FSMC_D4
  PE10   ------> FSMC_D7
  PE12   ------> FSMC_D9
  PE15   ------> FSMC_D12
  */
  GPIO_InitStruct.Pin = A19_Pin | FSMC_NBL1_Pin | FSMC_NBL0_Pin | A20_Pin | D10_Pin | D5_Pin | D6_Pin | D8_Pin | D11_Pin
                      | D4_Pin | D7_Pin | D9_Pin | D12_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF12_FSMC;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = FSMC_NE1_Pin | FSMC_NWAIT_Pin | D2_Pin | FSMC_NWE_Pin | D3_Pin | FSMC_NOE_Pin | FSMC_CLK_Pin
                      | D1_Pin | D0_Pin | A18_Pin | A17_Pin | A16_Pin | D15_Pin | D14_Pin | D13_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF12_FSMC;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = FSMC_NL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF12_FSMC;
  HAL_GPIO_Init(FSMC_NL_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = FSMC_NE3_Pin | FSMC_NE2_Pin | A15_Pin | A14_Pin | A13_Pin | A12_Pin | A11_Pin | A10_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF12_FSMC;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = A0_Pin | A2_Pin | A1_Pin | A3_Pin | A4_Pin | A5_Pin | A7_Pin | A6_Pin | A9_Pin | A8_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF12_FSMC;
  HAL_GPIO_Init(GPIOF, &GPIO_InitStruct);

  /* USER CODE BEGIN FSMC_MspInit 1 */

  /* USER CODE END FSMC_MspInit 1 */
}

void HAL_NOR_MspInit(NOR_HandleTypeDef* hnor) {
  /* USER CODE BEGIN NOR_MspInit 0 */

  /* USER CODE END NOR_MspInit 0 */
  HAL_FSMC_MspInit();
  /* USER CODE BEGIN NOR_MspInit 1 */

  /* USER CODE END NOR_MspInit 1 */
}

void HAL_SRAM_MspInit(SRAM_HandleTypeDef* hsram) {
  /* USER CODE BEGIN SRAM_MspInit 0 */

  /* USER CODE END SRAM_MspInit 0 */
  HAL_FSMC_MspInit();
  /* USER CODE BEGIN SRAM_MspInit 1 */

  /* USER CODE END SRAM_MspInit 1 */
}

static uint32_t FSMC_DeInitialized = 0;

static void HAL_FSMC_MspDeInit(void) {
  /* USER CODE BEGIN FSMC_MspDeInit 0 */

  /* USER CODE END FSMC_MspDeInit 0 */
  if (FSMC_DeInitialized)
    return;
  FSMC_DeInitialized = 1;
  /* Peripheral clock enable */
  __HAL_RCC_FSMC_CLK_DISABLE();

  /** FSMC GPIO Configuration
  PE3   ------> FSMC_A19
  PE1   ------> FSMC_NBL1
  PE0   ------> FSMC_NBL0
  PD7   ------> FSMC_NE1
  PE4   ------> FSMC_A20
  PB7   ------> FSMC_NL
  PG10   ------> FSMC_NE3
  PD6   ------> FSMC_NWAIT
  PD0   ------> FSMC_D2
  PG9   ------> FSMC_NE2
  PD5   ------> FSMC_NWE
  PD1   ------> FSMC_D3
  PD4   ------> FSMC_NOE
  PD3   ------> FSMC_CLK
  PF0   ------> FSMC_A0
  PF2   ------> FSMC_A2
  PF1   ------> FSMC_A1
  PF3   ------> FSMC_A3
  PF4   ------> FSMC_A4
  PF5   ------> FSMC_A5
  PG5   ------> FSMC_A15
  PG4   ------> FSMC_A14
  PG3   ------> FSMC_A13
  PD15   ------> FSMC_D1
  PG2   ------> FSMC_A12
  PG1   ------> FSMC_A11
  PD14   ------> FSMC_D0
  PD13   ------> FSMC_A18
  PF13   ------> FSMC_A7
  PG0   ------> FSMC_A10
  PE13   ------> FSMC_D10
  PD12   ------> FSMC_A17
  PD11   ------> FSMC_A16
  PD10   ------> FSMC_D15
  PF12   ------> FSMC_A6
  PF15   ------> FSMC_A9
  PE8   ------> FSMC_D5
  PE9   ------> FSMC_D6
  PE11   ------> FSMC_D8
  PE14   ------> FSMC_D11
  PD9   ------> FSMC_D14
  PD8   ------> FSMC_D13
  PF14   ------> FSMC_A8
  PE7   ------> FSMC_D4
  PE10   ------> FSMC_D7
  PE12   ------> FSMC_D9
  PE15   ------> FSMC_D12
  */
  HAL_GPIO_DeInit(GPIOE,
                  A19_Pin | FSMC_NBL1_Pin | FSMC_NBL0_Pin | A20_Pin | D10_Pin | D5_Pin | D6_Pin | D8_Pin | D11_Pin
                    | D4_Pin | D7_Pin | D9_Pin | D12_Pin);

  HAL_GPIO_DeInit(GPIOD,
                  FSMC_NE1_Pin | FSMC_NWAIT_Pin | D2_Pin | FSMC_NWE_Pin | D3_Pin | FSMC_NOE_Pin | FSMC_CLK_Pin | D1_Pin
                    | D0_Pin | A18_Pin | A17_Pin | A16_Pin | D15_Pin | D14_Pin | D13_Pin);

  HAL_GPIO_DeInit(FSMC_NL_GPIO_Port, FSMC_NL_Pin);

  HAL_GPIO_DeInit(GPIOG, FSMC_NE3_Pin | FSMC_NE2_Pin | A15_Pin | A14_Pin | A13_Pin | A12_Pin | A11_Pin | A10_Pin);

  HAL_GPIO_DeInit(GPIOF, A0_Pin | A2_Pin | A1_Pin | A3_Pin | A4_Pin | A5_Pin | A7_Pin | A6_Pin | A9_Pin | A8_Pin);

  /* USER CODE BEGIN FSMC_MspDeInit 1 */

  /* USER CODE END FSMC_MspDeInit 1 */
}

void HAL_NOR_MspDeInit(NOR_HandleTypeDef* hnor) {
  /* USER CODE BEGIN NOR_MspDeInit 0 */

  /* USER CODE END NOR_MspDeInit 0 */
  HAL_FSMC_MspDeInit();
  /* USER CODE BEGIN NOR_MspDeInit 1 */

  /* USER CODE END NOR_MspDeInit 1 */
}

void HAL_SRAM_MspDeInit(SRAM_HandleTypeDef* hsram) {
  /* USER CODE BEGIN SRAM_MspDeInit 0 */

  /* USER CODE END SRAM_MspDeInit 0 */
  HAL_FSMC_MspDeInit();
  /* USER CODE BEGIN SRAM_MspDeInit 1 */

  /* USER CODE END SRAM_MspDeInit 1 */
}

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
