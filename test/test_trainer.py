from self_driving_car.trainer import Trainer


def main():


    del env

    trainer = Trainer(CNNPolicy, inputs, outputs, './agent/data.hdf5')

    trainer.fit(batch_size=32, epochs=10)

    trainer.save()


if __name__ == '__main__':
    main()
