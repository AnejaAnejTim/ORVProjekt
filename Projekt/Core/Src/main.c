/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
/* USER CODE END Includes */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define TS_CAL1   ((uint16_t*)0x1FFFF7B8)
#define TS_CAL2   ((uint16_t*)0x1FFFF7C2)
#define VREFINT_CAL ((uint16_t*)0x1FFFF7BA)

#define ESP_SSID "iPhone"
#define ESP_PASS "Lol12345"

volatile int read = 1;
/* USER CODE END PD */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
I2C_HandleTypeDef hi2c1;
SPI_HandleTypeDef hspi1;
TIM_HandleTypeDef htim3;
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_SPI1_Init(void);
static void MX_ADC1_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM3_Init(void);

/* USER CODE BEGIN PFP */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* USER CODE BEGIN Init */
  /* USER CODE END Init */

  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C1_Init();
  MX_SPI1_Init();
  MX_ADC1_Init();
  MX_USB_DEVICE_Init();
  MX_USART2_UART_Init();
  MX_TIM3_Init();

  /* USER CODE BEGIN 2 */
  HAL_ADCEx_Calibration_Start(&hadc1, ADC_SINGLE_ENDED);
  HAL_NVIC_SetPriority(TIM3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(TIM3_IRQn);
  HAL_TIM_Base_Start_IT(&htim3);

  char usb_buf[64];
  static uint8_t rx_buf[512];
  static uint16_t idx = 0;

  HAL_Delay(1000);
  uint8_t dummy;
  while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

  HAL_UART_Transmit(&huart2, (uint8_t*)"ATE0\r\n", 6, HAL_MAX_DELAY);
  HAL_Delay(500);
  while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

  HAL_UART_Transmit(&huart2, (uint8_t*)"AT+CWMODE=1\r\n", 13, HAL_MAX_DELAY);
  HAL_Delay(500);
  while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

  char join[128];
  snprintf(join, sizeof(join), "AT+CWJAP=\"%s\",\"%s\"\r\n", ESP_SSID, ESP_PASS);
  HAL_UART_Transmit(&huart2, (uint8_t*)join, strlen(join), HAL_MAX_DELAY);
  HAL_Delay(8000);
  while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

  HAL_UART_Transmit(&huart2, (uint8_t*)"AT+CIPMUX=1\r\n", 13, HAL_MAX_DELAY);
  HAL_Delay(1000);
  while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

  HAL_UART_Transmit(&huart2, (uint8_t*)"AT+CIPSERVER=1,80\r\n", 19, HAL_MAX_DELAY);
  HAL_Delay(1000);
  while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

  /* USER CODE END 2 */

  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */
    /* USER CODE BEGIN 3 */

    if(read == 1) {
      ADC_ChannelConfTypeDef sConfig = {0};

      sConfig.Channel = ADC_CHANNEL_TEMPSENSOR;
      sConfig.Rank = ADC_REGULAR_RANK_1;
      sConfig.SamplingTime = ADC_SAMPLETIME_181CYCLES_5;
      sConfig.SingleDiff = ADC_SINGLE_ENDED;
      HAL_ADC_ConfigChannel(&hadc1, &sConfig);
      HAL_ADC_Start(&hadc1);
      HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
      uint32_t temp_adc = HAL_ADC_GetValue(&hadc1);
      HAL_ADC_Stop(&hadc1);

      sConfig.Channel = ADC_CHANNEL_VREFINT;
      HAL_ADC_ConfigChannel(&hadc1, &sConfig);
      HAL_ADC_Start(&hadc1);
      HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);
      uint32_t vref_adc = HAL_ADC_GetValue(&hadc1);
      HAL_ADC_Stop(&hadc1);

      float temp_adc_corr = (float)temp_adc * (*VREFINT_CAL) / (float)vref_adc;
      float temp = ((temp_adc_corr - *TS_CAL1) * 80.0f) / (*TS_CAL2 - *TS_CAL1) + 30.0f;
      int temp_x100 = (int)(temp * 100.0f);

      int len = snprintf(usb_buf, sizeof(usb_buf), "%d.%02d",
                        temp_x100 / 100, abs(temp_x100 % 100));
      char cdc_buf[64];
      int cdc_len = snprintf(cdc_buf, sizeof(cdc_buf), "%s\r\n", usb_buf);
      CDC_Transmit_FS((uint8_t*)cdc_buf, cdc_len);

      read = 0;
    }

    uint8_t c;
    if (HAL_UART_Receive(&huart2, &c, 1, 10) == HAL_OK) {
      if (idx < sizeof(rx_buf) - 1) {
        rx_buf[idx++] = c;
      }

      if (c == '\n') {
        rx_buf[idx] = 0;

        if (strstr((char*)rx_buf, "ready") || strstr((char*)rx_buf, "rst cause") || strstr((char*)rx_buf, "ets ")) {
          HAL_Delay(1000);
          while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

          HAL_UART_Transmit(&huart2, (uint8_t*)"ATE0\r\n", 6, HAL_MAX_DELAY);
          HAL_Delay(500);
          while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

          HAL_UART_Transmit(&huart2, (uint8_t*)"AT+CWMODE=1\r\n", 13, HAL_MAX_DELAY);
          HAL_Delay(500);
          while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

          snprintf(join, sizeof(join), "AT+CWJAP=\"%s\",\"%s\"\r\n", ESP_SSID, ESP_PASS);
          HAL_UART_Transmit(&huart2, (uint8_t*)join, strlen(join), HAL_MAX_DELAY);
          HAL_Delay(5000);
          while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

          HAL_UART_Transmit(&huart2, (uint8_t*)"AT+CIPMUX=1\r\n", 13, HAL_MAX_DELAY);
          HAL_Delay(500);
          while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}

          HAL_UART_Transmit(&huart2, (uint8_t*)"AT+CIPSERVER=1,80\r\n", 19, HAL_MAX_DELAY);
          HAL_Delay(500);
          while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}
        }

        if (strstr((char*)rx_buf, "+IPD") && strstr((char*)rx_buf, "GET /")) {
          int link_id = 0;
          char *p = strstr((char*)rx_buf, "+IPD,");
          if (p) link_id = atoi(p + 5);

          char http_buf[256];
          int http_len;
          char cmd[64];


          if (strstr((char*)rx_buf, "GET /wifi?")) {
            char new_ssid[64] = {0};
            char new_pass[64] = {0};

            char *ssid_pos = strstr((char*)rx_buf, "ssid=");
            if (ssid_pos) {
              ssid_pos += 5;
              char *ssid_end = strchr(ssid_pos, '&');
              if (!ssid_end) ssid_end = strchr(ssid_pos, ' ');
              if (ssid_end) {
                int n = ssid_end - ssid_pos;
                if (n > 63) n = 63;
                memcpy(new_ssid, ssid_pos, n);
              }
            }

            char *pass_pos = strstr((char*)rx_buf, "pass=");
            if (pass_pos) {
              pass_pos += 5;
              char *pass_end = strchr(pass_pos, ' ');
              if (!pass_end) pass_end = strchr(pass_pos, '&');
              if (pass_end) {
                int n = pass_end - pass_pos;
                if (n > 63) n = 63;
                memcpy(new_pass, pass_pos, n);
              }
            }

            if (new_ssid[0]) {
              http_len = snprintf(http_buf, sizeof(http_buf),
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n"
                "{\"ok\":true,\"msg\":\"Switching WiFi\"}\r\n");
            } else {
              http_len = snprintf(http_buf, sizeof(http_buf),
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n"
                "{\"ok\":false,\"error\":\"missing ssid\"}\r\n");
            }

            snprintf(cmd, sizeof(cmd), "AT+CIPSEND=%d,%d\r\n", link_id, http_len);
            HAL_UART_Transmit(&huart2, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
            HAL_Delay(200);
            HAL_UART_Transmit(&huart2, (uint8_t*)http_buf, http_len, HAL_MAX_DELAY);
            HAL_Delay(200);

            snprintf(cmd, sizeof(cmd), "AT+CIPCLOSE=%d\r\n", link_id);
            HAL_UART_Transmit(&huart2, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
            HAL_Delay(200);

            if (new_ssid[0]) {
              HAL_UART_Transmit(&huart2, (uint8_t*)"AT+CIPSERVER=0\r\n", 16, HAL_MAX_DELAY);
              HAL_Delay(200);
              HAL_UART_Transmit(&huart2, (uint8_t*)"AT+CWQAP\r\n", 10, HAL_MAX_DELAY);
              HAL_Delay(500);

              while (1)
              {
                while (HAL_UART_Receive(&huart2, &dummy, 1, 2) == HAL_OK) {}


                snprintf(join, sizeof(join), "AT+CWJAP=\"%s\",\"%s\"\r\n", new_ssid, new_pass);
                HAL_UART_Transmit(&huart2, (uint8_t*)join, strlen(join), HAL_MAX_DELAY);


                uint32_t start = HAL_GetTick();
                uint16_t tmpIdx = 0;
                char tmpBuf[512];
                memset(tmpBuf, 0, sizeof(tmpBuf));

                while ((HAL_GetTick() - start) < 5000)
                {
                  uint8_t ch;
                  if (HAL_UART_Receive(&huart2, &ch, 1, 50) == HAL_OK)
                  {
                    if (tmpIdx < sizeof(tmpBuf) - 1) tmpBuf[tmpIdx++] = (char)ch;
                    tmpBuf[tmpIdx] = 0;


                    if (strstr(tmpBuf, "WIFI GOT IP") || strstr(tmpBuf, "\r\nOK\r\n"))
                    {
                      goto WIFI_JOINED;
                    }


                    if (strstr(tmpBuf, "FAIL") || strstr(tmpBuf, "ERROR") || strstr(tmpBuf, "WIFI DISCONNECT"))
                    {
                      break;
                    }
                  }
                }

                HAL_Delay(5000);
              }

              WIFI_JOINED:

              HAL_UART_Transmit(&huart2, (uint8_t*)"AT+CIPMUX=1\r\n", 13, HAL_MAX_DELAY);
              HAL_Delay(200);
              HAL_UART_Transmit(&huart2, (uint8_t*)"AT+CIPSERVER=1,80\r\n", 19, HAL_MAX_DELAY);
              HAL_Delay(200);
            }
          }
          else {

        	  http_len = snprintf(http_buf, sizeof(http_buf),
        	      "HTTP/1.1 200 OK\r\n"
        	      "Content-Type: application/json\r\n"
        	      "Connection: close\r\n"
        	      "\r\n"
        	      "{\"temp\":\"%s\"}\r\n",
        	      usb_buf);


            snprintf(cmd, sizeof(cmd), "AT+CIPSEND=%d,%d\r\n", link_id, http_len);
            HAL_UART_Transmit(&huart2, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
            HAL_Delay(200);
            HAL_UART_Transmit(&huart2, (uint8_t*)http_buf, http_len, HAL_MAX_DELAY);
            HAL_Delay(200);

            snprintf(cmd, sizeof(cmd), "AT+CIPCLOSE=%d\r\n", link_id);
            HAL_UART_Transmit(&huart2, (uint8_t*)cmd, strlen(cmd), HAL_MAX_DELAY);
          }
        }

        idx = 0;
      }
    }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_USB|RCC_PERIPHCLK_USART2
                              |RCC_PERIPHCLK_I2C1;
  PeriphClkInit.Usart2ClockSelection = RCC_USART2CLKSOURCE_PCLK1;
  PeriphClkInit.I2c1ClockSelection = RCC_I2C1CLKSOURCE_HSI;
  PeriphClkInit.USBClockSelection = RCC_USBCLKSOURCE_PLL_DIV1_5;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{
  ADC_MultiModeTypeDef multimode = {0};
  ADC_ChannelConfTypeDef sConfig = {0};

  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV4;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.Overrun = ADC_OVR_DATA_OVERWRITTEN;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  multimode.Mode = ADC_MODE_INDEPENDENT;
  if (HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multimode) != HAL_OK)
  {
    Error_Handler();
  }

  sConfig.Channel = ADC_CHANNEL_TEMPSENSOR;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.SamplingTime = ADC_SAMPLETIME_181CYCLES_5;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = 0x00201D2B;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }

  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_4BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_4;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 7;
  hspi1.Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
  hspi1.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{
  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 9999;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 7199;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  HAL_GPIO_WritePin(GPIOE, CS_I2C_SPI_Pin|LD4_Pin|LD3_Pin|LD5_Pin
                          |LD7_Pin|LD9_Pin|LD10_Pin|LD8_Pin
                          |LD6_Pin, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = DRDY_Pin|MEMS_INT3_Pin|MEMS_INT4_Pin|MEMS_INT1_Pin
                          |MEMS_INT2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = CS_I2C_SPI_Pin|LD4_Pin|LD3_Pin|LD5_Pin
                           |LD7_Pin|LD9_Pin|LD10_Pin|LD8_Pin
                           |LD6_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);
}

/* USER CODE BEGIN 4 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
  if(htim->Instance == TIM3)
  {
    read = 1;
    HAL_GPIO_TogglePin(GPIOE, LD3_Pin);
  }
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif /* USE_FULL_ASSERT */
