#ifndef I2C_H
#define I2C_H

#include "lamp_core.h"


/**
 * i2c peripheral.
 */
struct I2C_peripheral {
    volatile uint32_t DVSR; // 00
    volatile uint32_t WREG; // 04
    volatile uint32_t RREG; // 08
};

#define I2C_0 ((struct I2C_peripheral *)(0x00300000))

#define I2C_START_CMD   (0x00 << 8)
#define I2C_WRITE_CMD   (0x01 << 8)
#define I2C_READ_CMD    (0x02 << 8)
#define I2C_STOP_CMD    (0x03 << 8)
#define I2C_RESTART_CMD (0x04 << 8)


void i2c_set_freq(void);

int i2c_read_transaction(const uint8_t dev, uint8_t *bytes,
                         uint8_t byte_num, uint8_t restart);

int i2c_write_transaction(const uint8_t dev, uint8_t *bytes,
                          uint8_t byte_num, uint8_t restart);

#endif
