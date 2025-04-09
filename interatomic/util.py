import pyscme

SCME_COMMIT = "274aa6fa4881bcb662d12a8c80488fa103a55fd2"

def verify_version():
    assert pyscme.version.commit() == SCME_COMMIT

